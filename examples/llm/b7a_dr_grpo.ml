(* B7a. Dr. GRPO: two normalisations in GRPO that bias its gradient.

   GRPO divides a completion's advantage by the group's standard deviation and
   its loss by the completion's length. Dr. GRPO removes both: the advantage is
   the reward minus the group mean, and each completion's sum over tokens is
   divided by a constant, the generation budget.

   1. Difficulty: the standard deviation is small when a prompt is almost
      always or almost never solved, so those groups get larger weights.
   2. Length: with a per-completion mean, every token of a long completion
      weighs less. Wrong completions (negative advantage) are pushed down less
      when long, correct ones are pushed up more when short.

   The length experiment needs completions whose length does not affect the
   reward: Lm_task.addition_with_filler lets up to 4 filler tokens "_" follow
   the answer, and the verifier ignores them.

   OPENBLAS_NUM_THREADS=1 dune exec examples/llm/b7a_dr_grpo.exe *)
open Owl_rllib
open Owl_rllib_llm
open Common

let task = Lm_task.addition_with_filler

(* Accuracy, and the mean filler of correct and of wrong completions. The
   digits are sampled, 8 per test problem from a fixed stream; the filler
   that follows them is not sampled but computed from the model:
   E[filler] = sum over j of P(at least j filler tokens). Sampling it as well
   would add noise larger than the effects measured. *)
let filler net =
  let problems = Array.concat (List.init 8 (fun _ -> test)) in
  let samples =
    R.generate ~max_len:task.max_len ~rng:(Random.State.make [| 0 |]) net
      (Array.map (fun (p : Lm_task.problem) -> p.prompt) problems)
  in
  let digits =
    Array.map
      (fun (s : R.sample) ->
        let n = ref 0 in
        while !n < Array.length s.tokens && s.tokens.(!n) < 10 do incr n done;
        Array.sub s.tokens 0 !n)
      samples
  in
  let alive = Array.make (Array.length samples) 1. and expected = Array.make (Array.length samples) 0. in
  for j = 0 to Lm_task.max_filler - 1 do
    let contexts = Array.mapi (fun i (s : R.sample) -> Array.concat [ s.prompt; digits.(i); Array.make j Lm_task.pad ]) samples in
    let logp = R.Graph.run (R.features contexts) net |> R.Distribution.Categorical.of_output |> R.Algodiff.unpack_arr in
    Array.iteri
      (fun i _ ->
        alive.(i) <- alive.(i) *. exp (R.Algodiff.A.get logp [| i; Lm_task.pad |] |> R.Algodiff.A.elt_to_float);
        expected.(i) <- expected.(i) +. alive.(i))
      samples
  done;
  let correct = ref [] and wrong = ref [] in
  Array.iteri
    (fun i (p : Lm_task.problem) ->
      if Array.append digits.(i) [| Lm_task.eos |] = p.answer then correct := expected.(i) :: !correct
      else wrong := expected.(i) :: !wrong)
    problems;
  let m l = Lm_rl.mean (Array.of_list l) in
  float_of_int (List.length !correct) /. float_of_int (Array.length problems), m !correct, m !wrong

(* move the weights by [eta] along the normalised negative gradient of [loss] *)
let step_along net loss eta =
  let ws, gs = R.Graph.backward net loss in
  let sq = ref 0. in
  Array.iter (Array.iter (fun g -> Array.iter (fun x -> sq := !sq +. (x *. x)) (R.values (R.Algodiff.unpack_arr g)))) gs;
  let c = eta /. sqrt !sq in
  R.Graph.update net (Array.map2 (Array.map2 (fun w g -> R.Algodiff.Maths.(w - (R.scalar c * g)))) ws gs)

let () =
  let iterations = ref 250 in
  Arg.parse [ "--iterations", Arg.Set_int iterations, "Training iterations" ] (fun _ -> ()) "B7a";

  (* 1. Difficulty bias, exactly: the total |advantage| of a group of 8 *)
  print_endline "Total |advantage| of a group of 8 completions, k of them correct:";
  Printf.printf "  %2s %12s %16s %8s\n" "k" "GRPO (std)" "Dr. GRPO (mean)" "ratio";
  for k = 1 to 7 do
    let rewards = Array.init 8 (fun i -> if i < k then 1. else 0.) in
    let total a = Array.fold_left (fun s x -> s +. Float.abs x) 0. a in
    let g = total (Lm_rl.grpo ~group_size:8 rewards) and d = total (Lm_rl.centred ~group_size:8 rewards) in
    Printf.printf "  %2d %12.3f %16.3f %8.2f\n" k g d (g /. d)
  done;
  print_endline "Relative to Dr. GRPO, GRPO weighs the groups with 1 or 7 correct 1.5 times as much as those with 4.\n";

  (* 2. Length bias: per-token weights in one group *)
  let start = fine_tuned ~task () in
  let a0, c0, w0 = filler start in
  Printf.printf "Fine-tuned with random filler: accuracy %.3f, mean filler %.2f (correct) %.2f (wrong)\n" a0 c0 w0;
  let batch = R.rollout ~task ~rngs:(Rng.create 3) ~prompts:1 ~group_size:8 ~problems:train start in
  let advantages = Lm_rl.grpo ~group_size:8 batch.rewards and centred = Lm_rl.centred ~group_size:8 batch.rewards in
  Printf.printf "%s, per-token weight of each completion (advantage x row weight, x 8):\n"
    (Lm_task.to_string batch.problems.(0).prompt);
  Array.iteri
    (fun i (s : R.sample) ->
      let n = float_of_int (Array.length s.tokens) in
      Printf.printf "  %-9s reward %.0f   GRPO %+.3f   Dr. GRPO %+.3f\n" (Lm_task.to_string s.tokens) batch.rewards.(i)
        (advantages.(i) /. n) (centred.(i) /. float_of_int task.max_len))
    batch.samples;

  (* 3. The bias in one gradient step. All aggregations see the same 32768
     completions; their noise is shared, so the differences are the bias. *)
  print_endline "\nOne normalised gradient step of length 0.02 on the same 4096 x 8 completions:";
  Printf.printf "  %-28s %8s %16s %14s\n" "aggregation" "accuracy" "filler: correct" "wrong";
  let big = R.rollout ~task ~rngs:(Rng.create 4) ~prompts:4096 ~group_size:8 ~problems:train start in
  let changes =
    List.map
      (fun (name, std, aggregation) ->
        let a = if std then Lm_rl.grpo ~group_size:8 big.rewards else Lm_rl.centred ~group_size:8 big.rewards in
        let weights = Lm_rl.row_weights aggregation ~lengths:big.lengths big.sequence in
        let net = R.Graph.copy start in
        let logits, _ = R.Graph.forward net (R.features big.contexts) in
        step_along net (R.pg_loss ~log_prob:(R.token_log_probs logits big.actions) ~advantages:(R.per_row a big) ~weights) 0.02;
        let acc, c, w = filler net in
        Printf.printf "  %-28s %+8.4f %+16.4f %+14.4f\n%!" name (acc -. a0) (c -. c0) (w -. w0);
        c, w)
      [ "GRPO (std, per-sequence mean)", true, Lm_rl.Sequence_mean; "token mean (std)", true, Lm_rl.Token_mean
      ; "Dr. GRPO", false, Lm_rl.Fixed_length task.max_len ]
  in
  let (gc, gw), (tc, tw) = List.nth changes 0, List.nth changes 1 in
  Printf.printf "GRPO minus token mean: filler of correct completions %+.4f, of wrong ones %+.4f.\n" (gc -. tc) (gw -. tw);
  print_endline "Any movement shared by all three comes from the batch, not from the aggregation.\n";

  (* 4. Training *)
  let base = { grpo_base with task } in
  List.iter
    (fun (label, config) ->
      let policy, _ = train_variant ~iterations:!iterations label config start in
      let _, c, w = filler policy in
      Printf.printf "  mean filler: correct %.2f, wrong %.2f\n" c w)
    [ "GRPO", base; "Dr. GRPO", R.dr_grpo base ]
