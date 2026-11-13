(* Shared by the B chapters: data, the fine-tuned starting point, evaluation. *)
open Owl_rllib
open Owl_rllib_llm
module R = Lm_rl.S

let train, test = Lm_task.split ~seed:0 ()

(* B0 stops fine-tuning here, when the model answers about 40% of the test
   problems with greedy decoding, so that RL has room to improve it *)
let sft_steps = 500

let fine_tuned ?(steps = sft_steps) ?task () =
  let net = R.create ~seed:1 () in
  R.sft ?task ~steps ~rng:(Random.State.make [| 1 |]) net train;
  net

(* sampled: one completion per test problem, the quantity RL maximises;
   greedy: the most likely token at every step *)
let print_eval label net =
  Printf.printf "%-34s test accuracy  sampled %.3f  greedy %.3f\n%!" label
    (R.accuracy ~sample:true net test) (R.accuracy net test)

let report ~every net i (s : R.stats) =
  if i mod every = 0 then
    Printf.printf "  iteration %4d  batch reward %.3f  KL to reference %.4f  test sampled %.3f  greedy %.3f\n%!" i
      s.reward s.kl (R.accuracy ~sample:true net test) (R.accuracy net test)

let show_completions ?(n = 6) ~rng net =
  let problems = Array.sub test 0 n in
  let samples = R.generate ~rng net (Array.map (fun (p : Lm_task.problem) -> p.prompt) problems) in
  Array.iter2
    (fun (p : Lm_task.problem) (s : R.sample) ->
      Printf.printf "  %s%-5s reward %.0f   (log pi = %s)\n" (Lm_task.to_string p.prompt) (Lm_task.to_string s.tokens)
        (Lm_task.outcome_reward p s.tokens)
        (String.concat " + " (Array.to_list (Array.map (Printf.sprintf "%.2f") s.log_probs))))
    problems samples

(* Sum over parameters of the variance of a gradient estimate across batches:
   the trace of its covariance. *)
let variance grads =
  let n = float_of_int (List.length grads) in
  let dim = Array.length (List.hd grads) in
  let mean = Array.make dim 0. in
  List.iter (Array.iteri (fun j g -> mean.(j) <- mean.(j) +. (g /. n))) grads;
  let var = ref 0. in
  List.iter (Array.iteri (fun j g -> var := !var +. (((g -. mean.(j)) ** 2.) /. (n -. 1.)))) grads;
  !var

let sample_batch ~rngs policy =
  R.rollout ~rngs ~prompts:R.pg_default.prompts ~group_size:R.pg_default.group_size ~problems:train policy

(* REINFORCE gradient for given per-completion advantages *)
let pg_gradient policy (batch : R.batch) advantages =
  let weights = Lm_rl.row_weights Lm_rl.Sequence_sum ~lengths:batch.lengths batch.sequence in
  let logits, _ = R.Graph.forward policy (R.features batch.contexts) in
  R.flat_gradient policy (R.pg_loss ~log_prob:(R.token_log_probs logits batch.actions) ~advantages:(R.per_row advantages batch) ~weights)

(* Every estimator sees the same batches from the same frozen policy. An
   estimator maps the rewards of a batch to one advantage per completion. *)
let compare_variance ?(batches = 32) estimators policy =
  let rngs = Rng.create 100 in
  let grads = List.map (fun _ -> ref []) estimators in
  let rewards = ref 0. in
  for _ = 1 to batches do
    let batch = sample_batch ~rngs policy in
    rewards := !rewards +. (Lm_rl.mean batch.rewards /. float_of_int batches);
    List.iter2 (fun (_, advantages) acc -> acc := pg_gradient policy batch (advantages batch.rewards) :: !acc) estimators grads
  done;
  Printf.printf "  %d batches of %d completions, mean reward %.3f\n" batches
    (R.pg_default.prompts * R.pg_default.group_size) !rewards;
  Printf.printf "  %-24s %16s\n" "estimator" "variance (trace)";
  let base = ref None in
  List.iter2
    (fun (name, _) acc ->
      let var = variance !acc in
      let b = match !base with None -> base := Some var; var | Some b -> b in
      Printf.printf "  %-24s %16.4g   (%.2f x the first)\n%!" name var (var /. b))
    estimators grads

(* |sum_t grad log pi(y_t)|^2 for every completion of a batch, one completion at a time *)
let score_norms policy (batch : R.batch) =
  Array.mapi
    (fun i _ ->
      let rows = List.filter (fun r -> batch.sequence.(r) = i) (List.init (Array.length batch.sequence) Fun.id) |> Array.of_list in
      let ones = Array.make (Array.length rows) 1. in
      let sub = { batch with contexts = Array.map (Array.get batch.contexts) rows; actions = Array.map (Array.get batch.actions) rows
                ; sequence = Array.make (Array.length rows) 0; lengths = [| Array.length rows |] } in
      pg_gradient policy sub ones |> Array.fold_left (fun a g -> a +. (g *. g)) 0.)
    batch.rewards

(* The constant baseline with the smallest variance is
   b* = E[R |score|^2] / E[|score|^2]: completions with large score norms
   dominate the noise, so their rewards matter most. *)
let optimal_baseline policy =
  let batch = sample_batch ~rngs:(Rng.create 50) policy in
  let s2 = score_norms policy batch in
  let by r = Lm_rl.mean (Array.of_list (List.filter_map (fun i -> if batch.rewards.(i) = r then Some s2.(i) else None) (List.init (Array.length s2) Fun.id))) in
  let b = Array.fold_left ( +. ) 0. (Array.map2 ( *. ) batch.rewards s2) /. Array.fold_left ( +. ) 0. s2 in
  Printf.printf "  mean |score|^2: correct completions %.0f, wrong ones %.0f; b* = %.3f (mean reward %.3f)\n" (by 1.) (by 0.) b
    (Lm_rl.mean batch.rewards);
  b

(* --- B7 --- *)

(* GRPO as the four variant papers run it: without a KL term *)
let grpo_base = { R.grpo_default with kl_coef = 0. }

let report_variant ~every ~task net i (s : R.stats) =
  if i mod every = 0 then
    Printf.printf "  iteration %4d  reward %.3f  entropy %.3f  zero-advantage groups %.2f  clipped %.3f  test sampled %.3f\n%!" i
      s.reward s.entropy s.zero_groups s.clipped (R.accuracy ~sample:true ~task net test)

let train_variant ?(iterations = 250) ?(every = 50) label (config : R.grpo_config) start =
  let policy = R.Graph.copy start in
  Printf.printf "%s\n%!" label;
  let t0 = Unix.gettimeofday () in
  let stats =
    R.train_grpo ~config ~reference:start ~iterations ~rngs:(Rng.create 7) ~problems:train policy
      ~report:(report_variant ~every ~task:config.task policy)
  in
  Printf.printf "  test accuracy sampled %.3f greedy %.3f; %d completions generated, %d gradient steps, %.0f s\n%!"
    (R.accuracy ~sample:true ~task:config.task policy test) (R.accuracy ~task:config.task policy test)
    (Array.fold_left (fun a (s : R.stats) -> a + s.completions) 0 stats)
    (iterations * config.epochs) (Unix.gettimeofday () -. t0);
  policy, stats
