(* Batched distributions: parameters [batch; action_dim], reduced results
   [batch; 1]. Categorical actions are one-hot arrays. Samples are constants. *)
module Make (AD : Owl_algodiff_generic_sig.Sig) = struct
  open AD
  let scalar x = F (A.float_to_elt x)
  let log_softmax x = Maths.(x - log_sum_exp ~axis:1 ~keep_dims:true x)
  let sum x = Maths.sum ~axis:1 ~keep_dims:true x

  module Categorical = struct
    type t = AD.t
    let of_output = log_softmax
    let log_prob logp actions = sum Maths.(logp * actions)
    let entropy logp = Maths.neg (sum Maths.(exp logp * logp))
    let kl p q = sum Maths.(exp p * (p - q))
    let sample ~rng logp =
      let p = unpack_arr (Maths.exp (primal' logp)) in
      let shape = A.shape p in
      let out = A.zeros shape in
      for i = 0 to shape.(0) - 1 do
        let probs = List.init shape.(1) (fun j -> j, A.get p [| i; j |] |> A.elt_to_float) in
        let j = Mdp.sample_from rng probs in
        A.set out [| i; j |] (A.float_to_elt 1.)
      done;
      out
  end

  module Gaussian = struct
    type t = { mean : AD.t; log_std : AD.t }
    let create ~mean ~log_std = { mean; log_std }
    let log_prob p action =
      let half = scalar 0.5 and c = scalar (0.5 *. log (2. *. Float.pi)) in
      sum Maths.(neg (half * sqr ((action - p.mean) / exp p.log_std) + p.log_std + c))
    let entropy p =
      (* zeros_like through mean also broadcasts a shared [1; dim] log_std. *)
      let c = scalar (0.5 *. log (2. *. Float.pi *. exp 1.)) in
      sum Maths.(p.mean * scalar 0. + p.log_std + c)
    let kl p q =
      sum Maths.(q.log_std - p.log_std
                 + (exp (scalar 2. * p.log_std) + sqr (p.mean - q.mean))
                   / (scalar 2. * exp (scalar 2. * q.log_std)) - scalar 0.5)
    let sample ~rng p =
      let mean = unpack_arr (primal' p.mean) in
      let noise = A.init (A.shape mean) (fun _ -> A.float_to_elt (Rng.normal rng)) in
      unpack_arr Maths.(primal' p.mean + exp (primal' p.log_std) * Arr noise)
  end
end
module S = Make (Owl.Neural.S.Algodiff)
module D = Make (Owl.Neural.D.Algodiff)
