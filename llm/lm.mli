(** A toy autoregressive language model.

    An MLP reads the last [window] tokens, one-hot encoded, and outputs logits for the next token
    (one-hot input followed by a linear layer is a learned embedding per position). In a batch,
    every generated position is one row. *)

module Make (Graph : Owl_neural_graph_sig.Sig) : sig
  include module type of struct
    include Neural.Make (Graph)
  end

  val vocab : int

  val window : int
  (** Context length: the longest prompt plus every answer token but the last. *)

  val input_size : int

  val encode : int array -> float array
  (** A context as its left-padded one-hot window, row-major [[position; token]]. *)

  val features : int array array -> Algodiff.t
  (** Encode a batch of contexts. *)

  val create : ?hidden:int list -> ?outputs:int -> seed:int -> unit -> Graph.network
  (** Hidden layers default to [[128; 128]]; [outputs] (default [vocab]) can be 1 for a critic. *)

  val token_log_probs : Algodiff.t -> int array -> Algodiff.t
  (** [log pi(action | context)] for every row, differentiable in the logits. *)

  type sample = {
    prompt : int array;
    tokens : int array;  (** generated tokens, ending with [eos] unless cut off *)
    log_probs : float array;  (** [log pi(token)] under the generating network *)
  }

  val generate :
    ?greedy:bool ->
    ?max_len:int ->
    rng:Random.State.t ->
    Graph.network ->
    int array array ->
    sample array
  (** Generate up to [max_len] tokens for every prompt (default
      {!Owl_rllib_llm.Lm_task.max_answer}), one batched forward pass per position; sampling by
      default, the most likely token with [~greedy:true]. *)

  val rows : (int array * int array) array -> int array array * int array * int array
  (** For (prompt, tokens) pairs, one row per token: the context before it, the token, and the index
      of its pair. *)

  val accuracy :
    ?sample:bool -> ?task:Lm_task.task -> Graph.network -> Lm_task.problem array -> float
  (** The fraction of problems answered right, with greedy decoding or one sample per problem; the
      sampling stream is fixed, so evaluations repeat. *)

  val sft :
    ?batch_size:int ->
    ?learning_rate:float ->
    ?report:(int -> float -> unit) ->
    ?task:Lm_task.task ->
    steps:int ->
    rng:Random.State.t ->
    Graph.network ->
    Lm_task.problem array ->
    unit
  (** Supervised fine-tuning with teacher forcing: minimise the cross-entropy of every target token
      given the tokens before it. [report step loss] is called after every step. *)
end

module S : module type of Make (Owl.Neural.S.Graph)
module D : module type of Make (Owl.Neural.D.Graph)
