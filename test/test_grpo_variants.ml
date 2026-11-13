open Owl_rllib
open Owl_rllib_llm
module R = Lm_rl.D
module AD = R.Algodiff

let reference =
  let tbl = Hashtbl.create 20 in
  let ic = open_in "fixtures/grpo_variants_numerics.txt" in
  (try
     while true do
       match input_line ic |> String.split_on_char ' ' with
       | key :: xs -> Hashtbl.add tbl key (Array.of_list (List.map float_of_string xs))
       | _ -> assert false
     done
   with End_of_file -> close_in ic);
  tbl

let compare name xs = Alcotest.(check (array (float 1e-9))) name (Hashtbl.find reference name) xs
let flat x = AD.unpack_arr x |> Owl.Dense.Ndarray.D.to_array
let train, _ = Lm_task.split ~seed:0 ()

(* the rows of test_m4: three completions of lengths 1, 2, 3 *)
let sequence = [| 0; 1; 1; 2; 2; 2 |]
let lengths = [| 1; 2; 3 |]
let logp0 = Array.map log [| 0.5; 0.9; 0.3; 0.7; 0.2; 0.6 |]
let old = Array.map log [| 0.4; 0.8; 0.5; 0.7; 0.1; 0.9 |]
let seq_adv = [| 1.5; -0.5; 2.0 |]
let adv = Array.map (fun s -> seq_adv.(s)) sequence

let advantages_and_weights () =
  compare "centred" (Lm_rl.centred ~group_size:4 [| 1.; 0.; 0.; 1.; 0.; 0.; 0.; 0.; 1.; 1.; 1.; 0. |]);
  compare "weights_fixed_length" (Lm_rl.row_weights (Lm_rl.Fixed_length 4) ~lengths sequence);
  compare "clip_fraction"
    [| Lm_rl.clip_fraction ~clip_low:0.2 ~clip_high:0.28 ~log_ratios:(Array.map2 ( -. ) logp0 old) ~advantages:adv |];
  compare "clip_fraction_negated"
    [| Lm_rl.clip_fraction ~clip_low:0.2 ~clip_high:0.28 ~log_ratios:(Array.map2 ( -. ) logp0 old)
         ~advantages:(Array.map Float.neg adv) |];
  Alcotest.(check (float 1e-12)) "zero groups" 0.5 (Lm_rl.zero_groups ~group_size:2 [| 1.; 1.; 0.; 1.; 0.; 0.; 1.; 0. |])

let losses () =
  let check name f =
    let lp = R.column logp0 in
    compare (name ^ "_loss") [| AD.unpack_flt (f lp) |];
    compare (name ^ "_grad") (flat (AD.grad f lp))
  in
  let old_log_prob = R.column old in
  check "clip_higher" (fun log_prob ->
      R.clipped_loss ~clip:0.2 ~clip_high:0.28 ~log_prob ~old_log_prob ~advantages:adv
        ~weights:(Lm_rl.row_weights Lm_rl.Sequence_mean ~lengths sequence));
  check "gspo" (fun log_prob ->
      R.sequence_clipped_loss ~clip:0.1 ~clip_high:0.2 ~log_prob ~old_log_prob ~advantages:seq_adv
        ~mean_matrix:(R.sequence_mean_matrix ~lengths sequence));
  check "cispo" (fun log_prob ->
      R.cispo_loss ~clip:0.2 ~clip_high:0.28 ~log_prob ~old_log_prob ~advantages:adv
        ~weights:(Lm_rl.row_weights Lm_rl.Token_mean ~lengths sequence))

let filler_task () =
  let task = Lm_task.addition_with_filler in
  let p = Lm_task.problem 27 48 in
  let reward s = task.reward p (Array.of_list (List.map (fun c -> match c with '$' -> Lm_task.eos | '_' -> Lm_task.pad | c -> Char.code c - 48) (List.of_seq (String.to_seq s)))) in
  Alcotest.(check (list (float 0.))) "filler is ignored" [ 1.; 1.; 0.; 0. ] (List.map reward [ "75$"; "75____$"; "75__"; "76_$" ]);
  let rng = Random.State.make [| 0 |] in
  let fillers = Hashtbl.create 5 in
  for _ = 1 to 200 do
    let t = task.target rng p in
    if task.reward p t <> 1. || Array.length t > task.max_len then Alcotest.fail "bad SFT target";
    Hashtbl.replace fillers (Array.length t) ()
  done;
  Alcotest.(check int) "every amount of filler" (Lm_task.max_filler + 1) (Hashtbl.length fillers)

let start () =
  let net = Lm_rl.S.create ~seed:1 () in
  Lm_rl.S.sft ~steps:300 ~rng:(Random.State.make [| 1 |]) net train;
  net

let dynamic_sampling () =
  let module S = Lm_rl.S in
  let policy = start () in
  let batch, first, completions =
    S.rollout_mixed ~task:Lm_task.addition ~rngs:(Rng.create 4) ~prompts:16 ~group_size:8 ~problems:train policy
  in
  let groups = Lm_rl.groups ~group_size:8 batch.rewards in
  Alcotest.(check int) "full batch" 16 (Array.length groups);
  Alcotest.(check (float 0.)) "no zero-advantage group" 0. (Lm_rl.zero_groups ~group_size:8 batch.rewards);
  Alcotest.(check bool) "resampled" true (Lm_rl.zero_groups ~group_size:8 first.rewards > 0. && completions > 128);
  Alcotest.(check int) "rows rebuilt" (Array.fold_left ( + ) 0 batch.lengths) (Array.length batch.actions);
  Alcotest.(check (array (float 1e-5))) "log-probs kept" (S.log_probs_of policy batch) batch.old_log_probs;
  (* an untrained network never answers correctly: every group is all zero *)
  let net = S.create ~seed:3 () in
  let config = S.dapo { S.grpo_default with prompts = 4 } in
  let stats = S.train_grpo ~config ~reference:(S.Graph.copy net) ~iterations:1 ~rngs:(Rng.create 0) ~problems:train net in
  Alcotest.(check int) "gives up after max_rounds" (S.max_rounds * 4 * 8) stats.(0).completions

(* One iteration of train_grpo is a few Adam steps on the loss each
   variant's paper prescribes, built here by hand from the same batch. A large
   learning rate moves the ratios away from 1, where clipping matters. *)
let wiring () =
  let fine_tuned = R.create ~seed:1 () in
  R.sft ~steps:100 ~rng:(Random.State.make [| 1 |]) fine_tuned train;
  let base = { R.grpo_default with prompts = 8; epochs = 3; kl_coef = 0.; learning_rate = 1e-2 } in
  let row_weights agg (b : R.batch) = Lm_rl.row_weights agg ~lengths:b.lengths b.sequence in
  let token_clip ~clip ~clip_high ~std agg (b : R.batch) log_prob =
    let a = if std then Lm_rl.grpo ~group_size:8 b.rewards else Lm_rl.centred ~group_size:8 b.rewards in
    R.clipped_loss ~clip ~clip_high ~log_prob ~old_log_prob:(R.column b.old_log_probs) ~advantages:(R.per_row a b)
      ~weights:(row_weights agg b)
  in
  List.iter
    (fun (name, (config : R.grpo_config), loss) ->
      let p = R.Graph.copy fine_tuned and q = R.Graph.copy fine_tuned in
      ignore (R.train_grpo ~config ~reference:fine_tuned ~iterations:1 ~rngs:(Rng.create 5) ~problems:train p);
      let rngs = Rng.create 5 in
      let batch =
        if config.dynamic_sampling then
          (fun (b, _, _) -> b) (R.rollout_mixed ~task:config.task ~rngs ~prompts:8 ~group_size:8 ~problems:train q)
        else R.rollout ~rngs ~prompts:8 ~group_size:8 ~problems:train q
      in
      let step = R.adam_clipped ~max_grad_norm:1. ~learning_rate:1e-2 q in
      for _ = 1 to 3 do
        let logits, _ = R.Graph.forward q (R.features batch.contexts) in
        step (loss batch (R.token_log_probs logits batch.actions))
      done;
      let weights n = R.Graph.mkpri n |> Array.to_list |> List.concat_map (fun l -> Array.to_list l |> List.map (fun w -> R.values (AD.unpack_arr w))) |> Array.concat in
      Alcotest.(check (array (float 1e-12))) name (weights q) (weights p))
    [ "grpo", base, token_clip ~clip:0.2 ~clip_high:0.2 ~std:true Lm_rl.Sequence_mean
    ; "dr grpo", R.dr_grpo base, token_clip ~clip:0.2 ~clip_high:0.2 ~std:false (Lm_rl.Fixed_length Lm_task.max_answer)
    ; "dapo", R.dapo base, token_clip ~clip:0.2 ~clip_high:0.28 ~std:true Lm_rl.Token_mean
    ; ( "gspo", R.gspo base
      , fun b log_prob ->
          R.sequence_clipped_loss ~clip:3e-4 ~clip_high:4e-4 ~log_prob ~old_log_prob:(R.column b.old_log_probs)
            ~advantages:(Lm_rl.grpo ~group_size:8 b.rewards) ~mean_matrix:(R.sequence_mean_matrix ~lengths:b.lengths b.sequence) )
    ; ( "cispo", R.cispo base
      , fun b log_prob ->
          R.cispo_loss ~clip:1. ~clip_high:0.2 ~log_prob ~old_log_prob:(R.column b.old_log_probs)
            ~advantages:(R.per_row (Lm_rl.grpo ~group_size:8 b.rewards) b) ~weights:(row_weights Lm_rl.Token_mean b) ) ]

let improves name (stats : Lm_rl.S.stats array) =
  let r a b = Lm_rl.mean (Array.map (fun (s : Lm_rl.S.stats) -> s.reward) (Array.sub stats a b)) in
  let n = Array.length stats in
  let first = r 0 5 and last = r (n - 5) 5 in
  if last < first +. 0.2 then Alcotest.failf "%s: reward %.3f -> %.3f" name first last

let learning () =
  let module S = Lm_rl.S in
  let small = Array.sub train 0 16 in
  let base = { S.grpo_default with prompts = 16; kl_coef = 0. } in
  List.iter
    (fun (name, config) ->
      let policy = start () in
      improves name
        (S.train_grpo ~config ~reference:(S.Graph.copy policy) ~iterations:40 ~rngs:(Rng.create 2) ~problems:small policy))
    [ "dr grpo", S.dr_grpo base; "dapo", S.dapo base; "gspo", S.gspo base; "cispo", S.cispo base ]

let () =
  Alcotest.run "GRPO variants"
    [ ( "numerics"
      , [ "advantages and weights", `Quick, advantages_and_weights; "losses", `Quick, losses; "filler task", `Quick, filler_task ] )
    ; "sampling", [ "dynamic sampling", `Quick, dynamic_sampling ]
    ; "configuration", [ "variants as switches", `Quick, wiring ]
    ; "learning", [ "dr grpo, dapo, gspo, cispo", `Slow, learning ] ]
