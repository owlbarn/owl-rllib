(** Stack the last observations along their last axis (the channel axis of NHWC images), so that a
    policy which sees one observation sees motion. *)

module Make (A : Owl_types_ndarray_algodiff.Sig) : sig
  val wrap : frames:int -> (A.arr, 'action) Ndenv.t -> (A.arr, 'action) Ndenv.t
  (** Oldest frame first. A reset fills the stack with copies of the first frame; the final
      observation of an episode ends with its true final frame. The Box bounds are tiled
      accordingly. Raises [Invalid_argument] unless [frames > 0] and observations are a Box. *)
end

module S : module type of Make (Owl.Neural.S.Algodiff.A)
module D : module type of Make (Owl.Neural.D.Algodiff.A)
