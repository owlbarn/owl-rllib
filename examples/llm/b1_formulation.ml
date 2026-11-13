(* B1. Language-model generation as an RL problem.

   Two views of the same thing. In the MDP view a state is the prompt plus the
   tokens so far, an action is one token, and the transition appends it. In
   the bandit view the action is the whole completion, with probability the
   product of its token probabilities. The reward comes from a verifier, either
   once at the end (outcome) or for every token (process).

   The second half trains the token MDP with the A6 REINFORCE code, unchanged.

   OPENBLAS_NUM_THREADS=1 dune exec examples/llm/b1_formulation.exe -- --critic --truncate *)
open Owl_rllib
open Owl_rllib_llm
open Common

let () =
  let process = ref false and critic = ref false and truncate = ref false and episodes = ref 32768 in
  Arg.parse
    [ "--process", Arg.Set process, "Train on process rewards instead of outcome rewards"
    ; "--critic", Arg.Set critic, "Learn a state-value baseline"
    ; "--truncate", Arg.Set truncate, "Report the length limit as a truncation (a mistake, for comparison)"
    ; "--episodes", Arg.Set_int episodes, "Training episodes" ]
    (fun _ -> ()) "B1";
  let policy = fine_tuned () in
  let p = Lm_task.problem 27 48 in
  let s = (R.generate ~rng:(Random.State.make [| 3 |]) policy [| p.prompt |]).(0) in

  (* 1. The MDP view, one step per token. *)
  Printf.printf "Prompt %s, sampled completion %s\n\nMDP view:\n" (Lm_task.to_string p.prompt) (Lm_task.to_string s.tokens);
  let process_r = Lm_task.process_rewards p s.tokens in
  Array.iteri
    (fun t token ->
      let state = Array.append p.prompt (Array.sub s.tokens 0 t) in
      let last = t = Array.length s.tokens - 1 in
      Printf.printf "  state %-10s action %s   pi(action | state) = %.3f   outcome reward %.0f   process reward %.3f\n"
        (Lm_task.to_string state) (Lm_task.to_string [| token |]) (exp s.log_probs.(t))
        (if last then Lm_task.outcome_reward p s.tokens else 0.) process_r.(t))
    s.tokens;

  (* 2. The bandit view: one action, one reward. *)
  let log_pi = Array.fold_left ( +. ) 0. s.log_probs in
  Printf.printf "\nBandit view: pi(%s | %s) = exp(%.3f) = %.4f, reward %.0f\n\n" (Lm_task.to_string s.tokens)
    (Lm_task.to_string p.prompt) log_pi (exp log_pi) (Lm_task.outcome_reward p s.tokens);

  (* 3. The same REINFORCE that balanced the pole, now writing answers. *)
  let reward = if !process then Lm_env.S.Process else Lm_env.S.Outcome in
  let env = Lm_env.S.make ~reward ~truncate_at_limit:!truncate ~rng:(Random.State.make [| 5 |]) train in
  let baseline = if !critic then Some (R.create ~seed:2 ~outputs:1 ()) else None in
  Printf.printf "A6 REINFORCE on the token MDP: %s reward, %s, length limit reported as %s\n"
    (if !process then "process" else "outcome") (if !critic then "learned baseline" else "no baseline")
    (if !truncate then "truncation" else "termination");
  print_eval "  before" policy;
  let chunk = !episodes / 4 in
  for i = 1 to 4 do
    ignore (Reinforce.S.train ?baseline ~gamma:1. ~batch_episodes:64 ~episodes:chunk ~rng:(Random.State.make [| i |]) ~policy env);
    print_eval (Printf.sprintf "  after %d episodes" (i * chunk)) policy
  done;
  print_endline "\nSampled completions after training:";
  show_completions ~rng:(Random.State.make [| 2 |]) policy
