(** Batched, differentiable action distributions.

    Parameters have shape [[batch; action_dim]]; [log_prob], [entropy] and [kl] return [[batch; 1]].
    Samples are constants (no gradient flows through them). *)

module Make (AD : Owl_algodiff_generic_sig.Sig) : sig
  (** A categorical distribution, represented by its log-probabilities. Actions are one-hot arrays.
  *)
  module Categorical : sig
    type t = AD.t

    val of_output : AD.t -> t
    (** Normalise logits with a log-softmax. *)

    val log_prob : t -> AD.t -> AD.t
    val entropy : t -> AD.t
    val kl : t -> t -> AD.t

    val sample : rng:Random.State.t -> t -> AD.A.arr
    (** One-hot samples. *)
  end

  (** A diagonal Gaussian. [log_std] may have shape [[1; action_dim]], shared by the whole batch. *)
  module Gaussian : sig
    type t = { mean : AD.t; log_std : AD.t }

    val create : mean:AD.t -> log_std:AD.t -> t
    val log_prob : t -> AD.t -> AD.t
    val entropy : t -> AD.t
    val kl : t -> t -> AD.t
    val sample : rng:Random.State.t -> t -> AD.A.arr
  end
end

module S : module type of Make (Owl.Neural.S.Algodiff)
module D : module type of Make (Owl.Neural.D.Algodiff)
