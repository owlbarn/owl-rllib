(** Token generation as an ordinary episodic environment (the MDP view), so that the classic
    algorithms apply directly.

    The state is the prompt plus the tokens generated so far, observed through the model's one-hot
    window; an action appends a token; the transition is deterministic. Each reset draws a new
    problem.

    An episode ends at [eos] or at the length limit, and both are terminal: a completion cut off at
    the limit is judged as it is. Reporting the limit as a truncation, as a Gym time limit would be,
    credits an unfinished completion with the critic's guess of rewards that never come;
    [~truncate_at_limit:true] reproduces that mistake for comparison. *)

module Make (Graph : Owl_neural_graph_sig.Sig) : sig
  type reward =
    | Outcome  (** the verifier's reward at the end *)
    | Process  (** {!Owl_rllib_llm.Lm_task.process_rewards}, one per token *)

  val make :
    ?reward:reward ->
    ?truncate_at_limit:bool ->
    rng:Random.State.t ->
    Lm_task.problem array ->
    (Graph.Neuron.Optimise.Algodiff.A.arr, int) Ndenv.t
end

module S : module type of Make (Owl.Neural.S.Graph)
module D : module type of Make (Owl.Neural.D.Graph)
