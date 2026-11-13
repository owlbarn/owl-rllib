(* B4. Staying close to the reference model, and reusing stale samples.

   1. KL(pi || pi_ref) per token is estimated from the sampled token alone:
      k1 = log pi - log pi_ref (unbiased, often negative), and
      k3 = r - log r - 1 with r = pi_ref / pi (unbiased, never negative).
      With a 14-token vocabulary we can also compute it exactly.
   2. A KL penalty in the reward trades reward for staying close.
   3. Importance sampling estimates the reward of a new policy from samples
      of an old one, weighting each completion by pi_new(y) / pi_old(y).

   OPENBLAS_NUM_THREADS=1 dune exec examples/llm/b4_kl.exe *)
open Owl_rllib
open Common

let std xs =
  let m = Lm_rl.mean xs in
  sqrt (Array.fold_left (fun a x -> a +. ((x -. m) ** 2.)) 0. xs /. float_of_int (Array.length xs - 1))

let () =
  let iterations = ref 250 in
  Arg.parse [ "--iterations", Arg.Set_int iterations, "Training iterations per KL coefficient" ] (fun _ -> ()) "B4";
  let reference = fine_tuned () in
  (* a policy that has moved away from the reference: 30 iterations of GRPO *)
  let moved = fine_tuned () in
  ignore (R.train_grpo ~reference ~iterations:30 ~rngs:(Rng.create 9) ~problems:train moved);

  (* 1. KL estimators on one batch sampled from the moved policy *)
  let batch = sample_batch ~rngs:(Rng.create 11) moved in
  let ref_lp = R.log_probs_of reference batch in
  let k1 = Array.map2 (fun log_prob ref_log_prob -> Lm_rl.k1 ~log_prob ~ref_log_prob) batch.old_log_probs ref_lp in
  let k3 = Array.map2 (fun log_prob ref_log_prob -> Lm_rl.k3 ~log_prob ~ref_log_prob) batch.old_log_probs ref_lp in
  let exact = R.exact_kl ~reference moved batch in
  let negative xs = float_of_int (Array.fold_left (fun a x -> if x < 0. then a + 1 else a) 0 xs) /. float_of_int (Array.length xs) in
  Printf.printf "KL(pi || pi_ref) per token, %d tokens sampled from a policy after 30 GRPO iterations:\n" (Array.length k1);
  Printf.printf "  %-26s %8s %10s %10s\n" "" "mean" "std" "negative";
  Printf.printf "  %-26s %8.4f %10.4f %10s\n" "exact (whole vocabulary)" (Lm_rl.mean exact) (std exact) "-";
  List.iter
    (fun (name, xs) -> Printf.printf "  %-26s %8.4f %10.4f %9.1f%%\n" name (Lm_rl.mean xs) (std xs) (100. *. negative xs))
    [ "k1 estimate", k1; "k3 estimate", k3 ];

  (* 2. The KL penalty: reward minus beta * (sum of k1 over the completion) *)
  List.iter
    (fun kl_coef ->
      Printf.printf "\nRLOO with KL penalty beta = %g, %d iterations:\n" kl_coef !iterations;
      let policy = fine_tuned () in
      let stats =
        R.train_pg ~config:{ R.pg_default with kl_coef } ~reference ~estimator:R.Leave_one_out ~iterations:!iterations
          ~rngs:(Rng.create 7) ~problems:train policy
      in
      let last = stats.(Array.length stats - 1) in
      Printf.printf "  last batch: reward %.3f, KL to reference %.4f per token\n" last.reward last.kl;
      print_eval "  after training" policy)
    [ 0.; 0.1; 1. ];

  (* 3. Importance sampling: the reward of [moved], from samples of [reference] *)
  let rngs = Rng.create 13 in
  let truth = Lm_rl.mean (Array.init 20 (fun _ -> Lm_rl.mean (sample_batch ~rngs moved).rewards)) in
  let estimates =
    Array.init 20 (fun _ ->
        let b = sample_batch ~rngs reference in
        let new_lp = R.log_probs_of moved b in
        let log_ratio = Array.make (Array.length b.rewards) 0. in
        Array.iteri (fun i s -> log_ratio.(s) <- log_ratio.(s) +. new_lp.(i) -. b.old_log_probs.(i)) b.sequence;
        let ratio = Array.map exp log_ratio in
        let weighted w = Lm_rl.mean (Array.map2 ( *. ) w b.rewards) in
        [| Lm_rl.mean b.rewards; weighted ratio; weighted (Array.map (Float.min 2.) ratio) |])
  in
  Printf.printf "\nMean reward of the moved policy: %.3f (from 20 batches of its own samples)\n" truth;
  Printf.printf "Estimated from 20 batches of reference samples, one estimate per batch:\n";
  List.iteri
    (fun j name ->
      let xs = Array.map (fun e -> e.(j)) estimates in
      Printf.printf "  %-34s mean %.3f  std %.3f\n" name (Lm_rl.mean xs) (std xs))
    [ "no correction"; "importance sampling"; "truncated importance sampling (2)" ]
