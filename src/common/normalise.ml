(* Observation and reward normalisation, as Gymnasium 0.29.1's
   NormalizeObservation and NormalizeReward (used by CleanRL's continuous
   PPO), each followed by an optional clip. Statistics are float64 running
   moments, updated with every value seen; each wrapped environment keeps
   its own, as with CleanRL's per-environment wrappers. *)

(* Running mean and variance, starting from mean 0, variance 1, count 1e-4. *)
type moments = { mean : float array; var : float array; mutable count : float }

let moments n = { mean = Array.make n 0.; var = Array.make n 1.; count = 1e-4 }

(* One sample: Chan et al.'s update with batch count 1 and batch variance 0. *)
let update m xs =
  let total = m.count +. 1. in
  Array.iteri (fun i x ->
    let delta = x -. m.mean.(i) in
    m.mean.(i) <- m.mean.(i) +. delta /. total;
    m.var.(i) <- (m.var.(i) *. m.count +. delta *. delta *. m.count /. total) /. total) xs;
  m.count <- total

let clip bound x = Float.max (-.bound) (Float.min bound x)

(* reward / sqrt (var + epsilon), where var is the running variance of the
   discounted return. As in Gymnasium, the return restarts only after a
   termination, not after a truncation. *)
let reward ?(epsilon = 1e-8) ?(bound = infinity) ~gamma (env : ('obs, 'action) Ndenv.t) : ('obs, 'action) Ndenv.t =
  let m = moments 1 and return = ref 0. in
  { env with
    step = (fun action ->
      let st = env.step action in
      return := !return *. gamma *. (if st.terminated then 0. else 1.) +. st.reward;
      update m [| !return |];
      { st with reward = clip bound (st.reward /. sqrt (m.var.(0) +. epsilon)) }) }

module Make (A : Owl_types_ndarray_algodiff.Sig) = struct
  let values x = let flat = A.reshape x [| A.numel x |] in Array.init (A.numel x) (fun i -> A.elt_to_float (A.get flat [| i |]))

  (* (obs - mean) / sqrt (var + epsilon), after adding obs to the statistics.
     The observation space becomes unbounded. *)
  let observation ?(epsilon = 1e-8) ?(bound = infinity) (env : (A.arr, 'action) Ndenv.t) : (A.arr, 'action) Ndenv.t =
    let shape = match env.observation_space with
      | Space.Box b -> b.shape
      | Discrete _ -> invalid_arg "Normalise.observation: Box observations required" in
    let n = Array.fold_left ( * ) 1 shape in
    let m = moments n in
    let normalise obs =
      let xs = values obs in
      update m xs;
      A.of_array (Array.mapi (fun i x -> A.float_to_elt (clip bound ((x -. m.mean.(i)) /. sqrt (m.var.(i) +. epsilon)))) xs) shape
    in
    { env with
      observation_space = Space.box ~low:(Array.make n (-.bound)) ~high:(Array.make n bound) ~shape
    ; reset = (fun () -> normalise (env.reset ()))
    ; step = (fun action -> let st = env.step action in { st with obs = normalise st.obs }) }
end

module S = Make (Owl.Neural.S.Algodiff.A)
module D = Make (Owl.Neural.D.Algodiff.A)
