(* Token generation as an ordinary episodic environment (the MDP view).

   The state is the prompt plus the tokens generated so far, observed through
   the model's one-hot window; an action appends one token; the transition is
   deterministic. Each reset draws a new problem.

   An episode ends at eos or at the length limit, and both are terminal: a
   completion cut off at the limit is judged as it is, nothing follows it.
   This differs from a Gym time limit, which is a truncation and must be
   bootstrapped; reporting the length limit as truncated would credit an
   unfinished completion with the critic's guess of rewards that never come.
   [~truncate_at_limit:true] reproduces that mistake for comparison. *)
module Make (Graph : Owl_neural_graph_sig.Sig) = struct
  module L = Lm.Make (Graph)

  type reward =
    | Outcome (* 1 at the end for a correct answer *)
    | Process (* credit for every correct token as it is generated *)

  let make ?(reward = Outcome) ?(truncate_at_limit = false) ~rng (problems : Lm_task.problem array) : int L.E.t =
    let observe (p, tokens) = L.encode (Array.append p.Lm_task.prompt (Array.of_list (List.rev tokens))) in
    let advance (p, tokens) token =
      let tokens = token :: tokens in
      let completion = Array.of_list (List.rev tokens) in
      let finished = token = Lm_task.eos || Array.length completion = Lm_task.max_answer in
      let r =
        match reward with
        | Outcome -> if finished then Lm_task.outcome_reward p completion else 0.
        | Process -> (Lm_task.process_rewards p completion).(Array.length completion - 1)
      in
      (p, tokens), r, if truncate_at_limit then token = Lm_task.eos else finished
    in
    L.E.simulator
      ~observation_space:(Space.box ~low:(Array.make L.input_size 0.) ~high:(Array.make L.input_size 1.) ~shape:[| L.input_size |])
      ~action_space:(Space.Discrete Lm_task.vocab) ~max_steps:(if truncate_at_limit then Lm_task.max_answer else Lm_task.max_answer + 1)
      ~reset_state:(fun () -> problems.(Random.State.int rng (Array.length problems)), [])
      ~observe ~advance
end

module S = Make (Owl.Neural.S.Graph)
module D = Make (Owl.Neural.D.Graph)
