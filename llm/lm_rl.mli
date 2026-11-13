(** RL for language models, following the progression REINFORCE, RLOO, PPO, GRPO and GRPO's variants
    (Dr. GRPO, DAPO, GSPO, CISPO).

    A batch holds [group_size] completions for each of several prompts, group-major: completion [i]
    belongs to prompt [i / group_size]. Losses are sums over token rows with one weight per row; the
    weights decide how per-token terms are aggregated ({!type-aggregation}). *)

(** {1 Advantages from outcome rewards, one per completion} *)

val mean : float array -> float

val groups : group_size:int -> 'a array -> 'a array array
(** Split into consecutive groups. Raises [Invalid_argument] unless the length is a multiple of
    [group_size]. *)

val batch_mean_baseline : float array -> float array
(** Rewards minus the batch mean. *)

val rloo : group_size:int -> float array -> float array
(** RLOO: each completion's baseline is the mean reward of the others in its group. *)

val grpo : ?eps:float -> group_size:int -> float array -> float array
(** GRPO: standardise within the group with the sample standard deviation ([eps] defaults to 1e-4).
*)

val centred : group_size:int -> float array -> float array
(** Dr. GRPO: only subtract the group mean. *)

val zero_groups : group_size:int -> 'a array -> float
(** The fraction of groups whose rewards are all equal (zero advantage). *)

val clip_fraction :
  clip_low:float -> clip_high:float -> log_ratios:float array -> advantages:float array -> float
(** The fraction of ratios outside [[1 - clip_low, 1 + clip_high]] on the side of their advantage,
    where the clipped surrogate has no gradient. *)

(** {1 KL estimators from one token sampled from [pi]} *)

val k1 : log_prob:float -> ref_log_prob:float -> float
(** Unbiased; can be negative. *)

val k3 : log_prob:float -> ref_log_prob:float -> float
(** Unbiased and never negative; usually lower variance when the policies are close. *)

(** {1 Aggregation} *)

type aggregation =
  | Sequence_sum  (** sum over a completion's tokens, mean over completions *)
  | Sequence_mean  (** mean over a completion's tokens, mean over completions *)
  | Token_mean  (** mean over all tokens in the batch *)
  | Fixed_length of int  (** Dr. GRPO: a completion's sum divided by a constant *)

val row_weights : aggregation -> lengths:int array -> int array -> float array
(** [row_weights a ~lengths sequence]: the weight of each row, given every completion's length and
    the completion each row belongs to. *)

(** {1 Training} *)

module Make (Graph : Owl_neural_graph_sig.Sig) : sig
  include module type of struct
    include Lm.Make (Graph)
  end

  type batch = {
    problems : Lm_task.problem array;  (** one per completion *)
    samples : sample array;
    rewards : float array;  (** outcome reward per completion *)
    lengths : int array;  (** tokens per completion *)
    contexts : int array array;  (** one per token row *)
    actions : int array;
    sequence : int array;  (** the completion of each row *)
    old_log_probs : float array;  (** [log pi_old], recorded while sampling *)
  }

  val batch_of : task:Lm_task.task -> Lm_task.problem array -> sample array -> batch

  val rollout :
    ?task:Lm_task.task ->
    rngs:Rng.t ->
    prompts:int ->
    group_size:int ->
    problems:Lm_task.problem array ->
    Graph.network ->
    batch
  (** Sample [group_size] completions for each of [prompts] random problems. *)

  val max_rounds : int

  val rollout_mixed :
    task:Lm_task.task ->
    rngs:Rng.t ->
    prompts:int ->
    group_size:int ->
    problems:Lm_task.problem array ->
    Graph.network ->
    batch * batch * int
  (** DAPO's dynamic sampling: sample batch after batch, keeping only groups whose rewards are not
      all equal, until [prompts] groups are kept or [max_rounds] batches have been sampled. Returns
      the kept groups, the first batch as sampled, and the number of completions generated. *)

  val log_probs_of : Graph.network -> batch -> float array
  (** [log pi(action)] of every row under a network, as constants. *)

  val exact_kl : reference:Graph.network -> Graph.network -> batch -> float array
  (** The exact [KL(pi || pi_ref)] over the vocabulary at every row. *)

  val per_row : 'a array -> batch -> 'a array
  (** Spread one value per completion over its rows. *)

  val sequence_means : float array -> batch -> float array
  (** The mean of per-row values over each completion's rows. *)

  (** {2 Losses, differentiable in [log_prob]} *)

  val pg_loss : log_prob:Algodiff.t -> advantages:float array -> weights:float array -> Algodiff.t

  val clip_ratio : clip:float -> clip_high:float -> Algodiff.t -> Algodiff.t
  (** Keep a ratio in [[1 - clip, 1 + clip_high]], differentiably. *)

  val clipped_loss :
    clip:float ->
    clip_high:float ->
    log_prob:Algodiff.t ->
    old_log_prob:Algodiff.t ->
    advantages:float array ->
    weights:float array ->
    Algodiff.t
  (** PPO's clipped surrogate with per-row weights; a larger [clip_high] is DAPO's clip-higher. *)

  val sequence_mean_matrix : lengths:int array -> int array -> Algodiff.t
  (** A [[n_seq; n_rows]] matrix whose product with a column of per-row values gives each
      completion's mean. *)

  val sequence_clipped_loss :
    clip:float ->
    clip_high:float ->
    log_prob:Algodiff.t ->
    old_log_prob:Algodiff.t ->
    advantages:float array ->
    mean_matrix:Algodiff.t ->
    Algodiff.t
  (** GSPO: one ratio per completion, the geometric mean of its token ratios, clipped; mean over
      completions. *)

  val cispo_loss :
    clip:float ->
    clip_high:float ->
    log_prob:Algodiff.t ->
    old_log_prob:Algodiff.t ->
    advantages:float array ->
    weights:float array ->
    Algodiff.t
  (** CISPO: the clipped ratio is a constant weight on [log pi], so a token whose ratio has left the
      range keeps a capped gradient. *)

  val k3_loss : log_prob:Algodiff.t -> ref_log_prob:Algodiff.t -> weights:float array -> Algodiff.t
  (** k3 as a loss term; its gradient pulls [pi] towards [pi_ref]. *)

  val flat_gradient : Graph.network -> Algodiff.t -> float array
  (** The gradient of a loss as one flat vector; the weights are unchanged. *)

  val log_ratios : log_prob:Algodiff.t -> batch -> float array
  (** [log pi_theta - log pi_old] of every row, as constants. *)

  val adam_clipped :
    max_grad_norm:float -> learning_rate:float -> Graph.network -> Algodiff.t -> unit
  (** [adam_clipped ~max_grad_norm ~learning_rate net] is an optimiser: a function that takes one
      clipped Adam step on [net] per loss. *)

  (** {2 Statistics} *)

  type stats = {
    reward : float;  (** mean outcome reward of the sampled completions *)
    kl : float;  (** mean exact KL to the reference per token, 0 without one *)
    length : float;  (** mean completion length *)
    entropy : float;  (** mean exact entropy per token *)
    zero_groups : float;  (** fraction of groups whose rewards are all equal *)
    completions : int;  (** completions generated in the iteration *)
    clipped : float;  (** fraction of rows (completions for GSPO) with a clipped ratio *)
  }

  val stats : ?reference:Graph.network -> group_size:int -> Graph.network -> batch -> stats
  (** Everything but [clipped], from a batch as sampled. *)

  (** {2 REINFORCE and RLOO: one gradient step per batch} *)

  type estimator = No_baseline | Batch_mean | Leave_one_out

  type pg_config = {
    prompts : int;
    group_size : int;
    learning_rate : float;
    max_grad_norm : float;
    kl_coef : float;  (** subtracted from the reward: [beta * sum of k1] over the completion *)
  }

  val pg_default : pg_config
  (** 64 prompts, 8 completions each, learning rate 1e-3, gradient norm 1, no KL. *)

  val pg_advantages :
    config:pg_config -> estimator:estimator -> ?reference:Graph.network -> batch -> float array

  val pg_gradient_loss :
    config:pg_config ->
    estimator:estimator ->
    ?reference:Graph.network ->
    Graph.network ->
    batch ->
    Algodiff.t

  val train_pg :
    ?config:pg_config ->
    ?reference:Graph.network ->
    ?report:(int -> stats -> unit) ->
    estimator:estimator ->
    iterations:int ->
    rngs:Rng.t ->
    problems:Lm_task.problem array ->
    Graph.network ->
    stats array
  (** Raises [Invalid_argument] if [kl_coef > 0] without a [reference]. *)

  (** {2 GRPO and its variants} *)

  (** How the policy term uses the ratio [pi_theta / pi_old]. *)
  type objective =
    | Token_clip  (** PPO's clipped surrogate on every token: GRPO, Dr. GRPO, DAPO *)
    | Sequence_clip  (** GSPO *)
    | Cispo  (** CISPO *)

  type grpo_config = {
    prompts : int;
    group_size : int;
    epochs : int;  (** gradient steps on each batch *)
    clip_low : float;  (** the ratio is kept in [[1 - clip_low, 1 + clip_high]] *)
    clip_high : float;
    kl_coef : float;  (** weight of the k3 term *)
    learning_rate : float;
    max_grad_norm : float;
    scale_by_std : bool;  (** divide centred rewards by the group's standard deviation *)
    aggregation : aggregation;
    objective : objective;
    dynamic_sampling : bool;  (** keep only groups whose rewards differ (DAPO) *)
    task : Lm_task.task;
  }

  val grpo_default : grpo_config
  (** GRPO: 64 prompts, 8 completions each, 4 epochs, clip 0.2, KL 0.1, learning rate 1e-3, gradient
      norm 1, standardised rewards, [Sequence_mean], [Token_clip], on
      {!Owl_rllib_llm.Lm_task.addition}. *)

  (** Each variant changes only the switches its paper changes. None of the papers uses a KL term:
      start from a configuration with [kl_coef = 0.] to match them. *)

  val dr_grpo : grpo_config -> grpo_config
  (** No division by the standard deviation or by the completion's length. *)

  val dapo : grpo_config -> grpo_config
  (** Clip-higher (0.2 / 0.28), a token-level mean and dynamic sampling. The paper's overlong reward
      shaping is not implemented. *)

  val gspo : grpo_config -> grpo_config
  (** Sequence-level ratios, with the paper's clipping range 3e-4 / 4e-4. *)

  val cispo : grpo_config -> grpo_config
  (** No lower bound on the weight and a token-level mean; [clip_high] is kept (the paper tunes it
      without giving a value). *)

  val train_grpo :
    ?config:grpo_config ->
    ?report:(int -> stats -> unit) ->
    reference:Graph.network ->
    iterations:int ->
    rngs:Rng.t ->
    problems:Lm_task.problem array ->
    Graph.network ->
    stats array
  (** Three policies: [pi_old] generated the batch, [pi_theta] (the network argument) is optimised,
      [reference] is the frozen starting point. *)

  (** {2 Token-level PPO with a critic} *)

  type ppo_config = {
    prompts : int;
    group_size : int;  (** completions per prompt, to match GRPO's sampling budget *)
    epochs : int;
    clip : float;
    kl_coef : float;  (** per-token penalty [-beta * k1] in the reward *)
    gamma : float;
    lambda : float;
    learning_rate : float;
    max_grad_norm : float;
  }

  val ppo_default : ppo_config
  (** 64 prompts, 8 completions each, 4 epochs, clip 0.2, KL 0.1, gamma 1, lambda 0.95, learning
      rate 1e-3, gradient norm 1. *)

  val token_advantages :
    config:ppo_config ->
    ref_log_probs:float array ->
    values:float array ->
    batch ->
    float array * float array
  (** GAE advantages and returns from token rewards: [-beta * k1] at every token plus the outcome
      reward at the last. Every completion ends in a terminal state, whether at [eos] or at the
      length limit. *)

  val train_ppo :
    ?config:ppo_config ->
    ?report:(int -> stats -> unit) ->
    reference:Graph.network ->
    critic:Graph.network ->
    iterations:int ->
    rngs:Rng.t ->
    problems:Lm_task.problem array ->
    Graph.network ->
    stats array
end

module S : module type of Make (Owl.Neural.S.Graph)
module D : module type of Make (Owl.Neural.D.Graph)
