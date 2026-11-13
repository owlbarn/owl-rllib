(* Adapt neural outputs and stored actions to either action space. The Gaussian
   stores the UNCLIPPED sample; only actions sent to the environment are clipped.
   Its log density is therefore that of the latent Gaussian, not a clipped PDF. *)
type ('tensor, 'array, 'action) t =
  { sample : rng:Random.State.t -> 'tensor -> 'array
  ; log_prob : 'tensor -> 'tensor -> 'tensor
  ; entropy : 'tensor -> 'tensor
  ; actions : 'array -> 'action array }

module Make (AD : Owl_algodiff_generic_sig.Sig) = struct
  open AD
  module Dist = Dist.Make (AD)
  let row matrix i =
    let dim = (A.shape matrix).(1) in
    A.init [| dim |] (fun j -> A.get matrix [| i; j |])
  let categorical =
    let distribution = Dist.Categorical.of_output in
    { sample = (fun ~rng output -> Dist.Categorical.sample ~rng (distribution output))
    ; log_prob = (fun output action -> Dist.Categorical.log_prob (distribution output) action)
    ; entropy = (fun output -> Dist.Categorical.entropy (distribution output))
    ; actions = (fun samples -> Array.init (A.shape samples).(0) (fun i ->
        let xs = Array.init (A.shape samples).(1) (fun j -> A.get samples [| i; j |] |> A.elt_to_float) in
        Dp.argmax xs)) }

  (* Network output [mean_1 ... mean_d log_std_1 ... log_std_d].
     State-dependent scales are bounded to keep exp(log_std) well-conditioned. *)
  let gaussian ~low ~high =
    let dim = Array.length low in
    if dim = 0 || Array.length high <> dim || not (Array.for_all2 ( < ) low high)
    then invalid_arg "Policy.gaussian: bounds";
    let low = Array.copy low and high = Array.copy high in
    let distribution output =
      let parts = Maths.split ~axis:1 [| dim; dim |] output in
      let log_std = Maths.(min2 (F (A.float_to_elt 2.))
          (max2 (F (A.float_to_elt (-5.))) parts.(1))) in
      Dist.Gaussian.create ~mean:parts.(0) ~log_std
    in
    { sample = (fun ~rng output -> Dist.Gaussian.sample ~rng (distribution output))
    ; log_prob = (fun output action -> Dist.Gaussian.log_prob (distribution output) action)
    ; entropy = (fun output -> Dist.Gaussian.entropy (distribution output))
    ; actions = (fun samples -> Array.init (A.shape samples).(0) (fun i ->
        A.init [| dim |] (fun j ->
          A.get samples [| i; j |] |> A.elt_to_float
          |> Float.min high.(j) |> Float.max low.(j) |> A.float_to_elt))) }
end
module S = Make (Owl.Neural.S.Algodiff)
module D = Make (Owl.Neural.D.Algodiff)
