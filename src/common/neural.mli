(** Networks and tensor conversions shared by the algorithms.

    [Make (Graph)] also re-exports Owl's neural network modules for [Graph] ([Graph], [Algodiff],
    [Activation], [Init], [Params], [Stepper], ...), so that one module gives access to everything
    an algorithm needs. *)

module Make (Graph : Owl_neural_graph_sig.Sig) : sig
  include module type of struct
    include Owl_neural_generic.Flatten (Graph)
  end

  module E : module type of struct
    include Ndenv.Make (Algodiff.A)
  end

  module B : module type of struct
    include Buffer.Make (Algodiff.A)
  end

  module Distribution : module type of struct
    include Dist.Make (Algodiff)
  end

  (** {1 Tensors} *)

  val scalar : float -> Algodiff.t

  val column : float array -> Algodiff.t
  (** A [[n; 1]] matrix. *)

  val batch : Algodiff.A.arr array -> Algodiff.t
  (** Stack observations along a new batch axis. *)

  val one_hot : int -> int array -> Algodiff.t
  (** [one_hot n actions] is a [[batch; n]] matrix. *)

  val values : Algodiff.A.arr -> float array
  (** All elements, in row-major order. *)

  val predict : Graph.network -> Algodiff.A.arr -> float array
  (** The network's output for one observation. *)

  val n_actions : (Algodiff.A.arr, int) Ndenv.t -> int
  (** Raises [Invalid_argument] unless actions are discrete. *)

  (** {1 Training} *)

  val adam : float -> Params.typ
  (** Adam with the given learning rate, betas 0.9 and 0.999. *)

  val update : Graph.network -> Stepper.state -> Algodiff.t -> unit
  (** One optimiser step on a loss computed from [Graph.forward]. *)

  (** {1 Networks}

      Every constructor seeds Owl's generator with [seed] before initialising the weights, so a seed
      always gives the same network. *)

  val mlp : seed:int -> shape:int array -> hidden:int list -> outputs:int -> Graph.network
  (** ReLU hidden layers and a linear output, with Owl's default initialisation. *)

  type init =
    | Default  (** Owl's default initialisation *)
    | Orthogonal of float
        (** orthogonal weights, gain [sqrt 2] for hidden layers and the given gain for the output
            layer (as in CleanRL's PPO) *)

  val orthogonal : float -> Init.typ
  (** Orthogonal weights scaled by a gain, like PyTorch's [orthogonal_]. *)

  val mlp_with :
    activation:Activation.typ ->
    init:init ->
    seed:int ->
    shape:int array ->
    hidden:int list ->
    outputs:int ->
    Graph.network

  val gaussian_mlp :
    activation:Activation.typ ->
    init:init ->
    seed:int ->
    shape:int array ->
    hidden:int list ->
    actions:int ->
    Graph.network
  (** A network for {!Owl_rllib.Policy.Make.gaussian} whose log_std does not depend on the state:
      one learned vector, initially 0. *)

  val cnn :
    seed:int ->
    shape:int array ->
    convs:(int * int * int) list ->
    hidden:int list ->
    outputs:int ->
    Graph.network
  (** For NHWC images [[height; width; channels]]: each [(kernel, stride, channels)] is a VALID ReLU
      convolution, followed by ReLU hidden layers on the flattened features and a linear output. *)
end
