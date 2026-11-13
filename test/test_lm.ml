open Owl_rllib
module R = Lm_rl.D
module AD = R.Algodiff

let reference =
  let tbl = Hashtbl.create 20 in
  let ic = open_in "fixtures/lm_numerics.txt" in
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
let train, test = Lm_task.split ~seed:0 ()

let task () =
  let p = Lm_task.problem 27 48 in
  Alcotest.(check string) "prompt" "27+48=" (Lm_task.to_string p.prompt);
  Alcotest.(check string) "answer" "75$" (Lm_task.to_string p.answer);
  Alcotest.(check (float 0.)) "correct" 1. (Lm_task.outcome_reward p p.answer);
  Alcotest.(check (float 0.)) "no eos" 0. (Lm_task.outcome_reward p [| 7; 5 |]);
  Alcotest.(check (array (float 1e-12))) "process credit" [| 1. /. 3.; 0.; 1. /. 3. |]
    (Lm_task.process_rewards p [| 7; 4; Lm_task.eos |]);
  Alcotest.(check int) "split covers every pair" 8100 (Array.length train + Array.length test);
  let seen = Hashtbl.create 8100 in
  Array.iter (fun (p : Lm_task.problem) -> Hashtbl.replace seen (p.a, p.b) ()) train;
  Alcotest.(check bool) "disjoint" true (Array.for_all (fun (p : Lm_task.problem) -> not (Hashtbl.mem seen (p.a, p.b))) test)

let encoding () =
  let x = R.encode [| 1; 2 |] in
  Alcotest.(check (float 0.)) "one token per position" (float_of_int R.window) (Array.fold_left ( +. ) 0. x);
  Alcotest.(check (float 0.)) "left padding" 1. x.(Lm_task.pad);
  Alcotest.(check (float 0.)) "last token" 1. x.(((R.window - 1) * R.vocab) + 2)

let advantages () =
  let rewards = [| 1.; 0.; 0.; 1.; 0.; 0.; 0.; 0.; 1.; 1.; 1.; 0. |] in
  compare "rloo" (Lm_rl.rloo ~group_size:4 rewards);
  compare "grpo" (Lm_rl.grpo ~group_size:4 rewards)

let rows_sequence = [| 0; 1; 1; 2; 2; 2 |]
let lengths = [| 1; 2; 3 |]
let logp0 = Array.map log [| 0.5; 0.9; 0.3; 0.7; 0.2; 0.6 |]
let old = Array.map log [| 0.4; 0.8; 0.5; 0.7; 0.1; 0.9 |]
let ref_lp = Array.map log [| 0.45; 0.6; 0.4; 0.8; 0.3; 0.5 |]
let adv = [| 1.5; -0.5; -0.5; 2.; 2.; 2. |]

let estimators_and_weights () =
  compare "k1" (Array.map2 (fun log_prob ref_log_prob -> Lm_rl.k1 ~log_prob ~ref_log_prob) logp0 ref_lp);
  compare "k3" (Array.map2 (fun log_prob ref_log_prob -> Lm_rl.k3 ~log_prob ~ref_log_prob) logp0 ref_lp);
  List.iter
    (fun (name, agg) -> compare ("weights_" ^ name) (Lm_rl.row_weights agg ~lengths rows_sequence))
    [ "sequence_sum", Lm_rl.Sequence_sum; "sequence_mean", Lm_rl.Sequence_mean; "token_mean", Lm_rl.Token_mean ]

let losses () =
  let weights = Lm_rl.row_weights Lm_rl.Sequence_mean ~lengths rows_sequence in
  let check name f =
    let lp = R.column logp0 in
    compare (name ^ "_loss") [| AD.unpack_flt (f lp) |];
    compare (name ^ "_grad") (flat (AD.grad f lp))
  in
  check "pg" (fun log_prob -> R.pg_loss ~log_prob ~advantages:adv ~weights);
  check "clipped" (fun log_prob ->
      R.clipped_loss ~clip:0.2 ~clip_high:0.2 ~log_prob ~old_log_prob:(R.column old) ~advantages:adv ~weights);
  check "k3" (fun log_prob -> R.k3_loss ~log_prob ~ref_log_prob:(R.column ref_lp) ~weights)

let token_advantages () =
  let batch =
    { R.problems = [||]; samples = [||]; rewards = [| 1.; 0.; 1. |]; lengths; contexts = [||]
    ; actions = Array.make 6 0; sequence = rows_sequence; old_log_probs = old }
  in
  let advantages, returns =
    R.token_advantages ~config:R.ppo_default ~ref_log_probs:ref_lp ~values:[| 0.3; 0.2; 0.1; 0.6; 0.5; 0.4 |] batch
  in
  compare "token_advantages" advantages;
  compare "token_returns" returns

let generation () =
  let net = R.create ~seed:3 () in
  let rngs = Rng.create 1 in
  let batch = R.rollout ~rngs ~prompts:8 ~group_size:4 ~problems:train net in
  Alcotest.(check int) "completions" 32 (Array.length batch.samples);
  Array.iter
    (fun (s : R.sample) ->
      let n = Array.length s.tokens in
      if not (n = Lm_task.max_answer || s.tokens.(n - 1) = Lm_task.eos) then Alcotest.fail "completion ends early")
    batch.samples;
  Alcotest.(check int) "one row per token" (Array.fold_left ( + ) 0 batch.lengths) (Array.length batch.actions);
  Alcotest.(check (array (float 1e-9))) "recorded log-probs" (R.log_probs_of net batch) batch.old_log_probs;
  Alcotest.(check (float 1e-12)) "no KL to itself" 0. (Lm_rl.mean (R.exact_kl ~reference:(R.Graph.copy net) net batch))

(* the mean of k1 and of k3 over tokens sampled from pi is KL(pi || pi_ref) *)
let kl_estimators () =
  let p = R.create ~seed:4 () and q = R.create ~seed:5 () in
  let context = [| [| 1; 2; Lm_task.plus; 3; 4; Lm_task.equals |] |] in
  let logp net = R.Graph.run (R.features context) net |> R.Distribution.Categorical.of_output |> AD.unpack_arr |> R.values in
  let lp = logp p and lq = logp q in
  let exact = Array.fold_left ( +. ) 0. (Array.mapi (fun v x -> exp x *. (x -. lq.(v))) lp) in
  let rng = Random.State.make [| 6 |] in
  let n = 50_000 in
  let k1 = ref 0. and k3 = ref 0. in
  for _ = 1 to n do
    let v = Mdp.sample_from rng (List.init R.vocab (fun v -> v, exp lp.(v))) in
    k1 := !k1 +. Lm_rl.k1 ~log_prob:lp.(v) ~ref_log_prob:lq.(v);
    k3 := !k3 +. Lm_rl.k3 ~log_prob:lp.(v) ~ref_log_prob:lq.(v)
  done;
  let close name x =
    if Float.abs ((x /. float_of_int n) -. exact) > 0.05 *. exact then
      Alcotest.failf "%s: estimate %.4f, exact %.4f" name (x /. float_of_int n) exact
  in
  close "k1" !k1;
  close "k3" !k3

let environment () =
  let p = Lm_task.problem 27 48 in
  let run ?truncate_at_limit ?reward tokens =
    let env = Lm_env.D.make ?truncate_at_limit ?reward ~rng:(Random.State.make [| 0 |]) [| p |] in
    ignore (env.reset ());
    List.map env.step tokens
  in
  let last steps = List.nth steps (List.length steps - 1) in
  let correct = last (run [ 7; 5; Lm_task.eos ]) in
  Alcotest.(check bool) "correct answer" true (correct.reward = 1. && correct.terminated && not correct.truncated);
  let cut = last (run [ 7; 5; 5; 5 ]) in
  Alcotest.(check bool) "length limit is terminal" true (cut.reward = 0. && cut.terminated && not cut.truncated);
  let wrong = last (run ~truncate_at_limit:true [ 7; 5; 5; 5 ]) in
  Alcotest.(check bool) "optional truncation" true (wrong.truncated && not wrong.terminated);
  let process = run ~reward:Lm_env.D.Process [ 7; 5; Lm_task.eos ] in
  Alcotest.(check (float 1e-12)) "process rewards sum to one" 1.
    (List.fold_left (fun a (st : _ Ndenv.step) -> a +. st.reward) 0. process)

(* --- learning on 16 problems, starting from a briefly fine-tuned model --- *)

let small = Array.sub train 0 16

let start () =
  let net = Lm_rl.S.create ~seed:1 () in
  Lm_rl.S.sft ~steps:300 ~rng:(Random.State.make [| 1 |]) net train;
  net

let improves name (stats : Lm_rl.S.stats array) =
  let r a b = Lm_rl.mean (Array.map (fun (s : Lm_rl.S.stats) -> s.reward) (Array.sub stats a b)) in
  let n = Array.length stats in
  let first = r 0 5 and last = r (n - 5) 5 in
  if last < first +. 0.2 then Alcotest.failf "%s: reward %.3f -> %.3f" name first last

let learning () =
  let module S = Lm_rl.S in
  let pg = { S.pg_default with prompts = 16 } in
  let policy = start () in
  improves "rloo" (S.train_pg ~config:pg ~estimator:S.Leave_one_out ~iterations:40 ~rngs:(Rng.create 2) ~problems:small policy);
  let policy = start () in
  let reference = S.Graph.copy policy in
  improves "grpo"
    (S.train_grpo ~config:{ S.grpo_default with prompts = 16 } ~reference ~iterations:40 ~rngs:(Rng.create 2) ~problems:small policy);
  let policy = start () in
  let critic = S.create ~seed:2 ~outputs:1 () in
  improves "ppo"
    (S.train_ppo ~config:{ S.ppo_default with prompts = 16 } ~reference ~critic ~iterations:40 ~rngs:(Rng.create 2)
       ~problems:small policy)

let deterministic () =
  let module S = Lm_rl.S in
  let run () =
    let policy = start () in
    let reference = S.Graph.copy policy in
    let stats =
      S.train_grpo ~config:{ S.grpo_default with prompts = 8 } ~reference ~iterations:5 ~rngs:(Rng.create 3) ~problems:train policy
    in
    stats, S.Graph.mkpri policy |> Array.map (Array.map (fun w -> S.values (S.Algodiff.unpack_arr w)))
  in
  Alcotest.(check bool) "same seed, same result" true (run () = run ())

let () =
  Alcotest.run "language models"
    [ ( "task"
      , [ "addition task", `Quick, task; "encoding", `Quick, encoding; "environment", `Quick, environment ] )
    ; ( "numerics"
      , [ "advantages", `Quick, advantages
        ; "estimators and weights", `Quick, estimators_and_weights
        ; "losses", `Quick, losses
        ; "token advantages", `Quick, token_advantages
        ; "generation", `Quick, generation
        ; "kl estimators", `Quick, kl_estimators ] )
    ; "learning", [ "rloo, grpo, ppo", `Slow, learning; "deterministic", `Quick, deterministic ] ]
