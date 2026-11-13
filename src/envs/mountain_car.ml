module Make (A : Owl_types_ndarray_algodiff.Sig) = struct
  module E = Ndenv.Make (A)
  let clip x lo hi = Float.max lo (Float.min hi x)

  let advance s action =
    if action < 0 || action > 2 then invalid_arg "Mountain_car: action";
    let v = clip (s.(1) +. float_of_int (action - 1) *. 0.001 -. cos (3. *. s.(0)) *. 0.0025) (-0.07) 0.07 in
    let x = clip (s.(0) +. v) (-1.2) 0.6 in
    let v = if x = -1.2 && v < 0. then 0. else v in
    [| x; v |], -1., x >= 0.5 && v >= 0.

  let make ?(max_steps = 200) ?initial_state ~rng () =
    let reset_state () = match initial_state with
      | Some s -> if Array.length s <> 2 then invalid_arg "Mountain_car: state" else Array.copy s
      | None -> [| -0.6 +. Random.State.float rng 0.2; 0. |]
    in
    E.simulator ~max_steps ~reset_state ~observe:Fun.id ~advance
      ~observation_space:(Space.box ~low:[| -1.2; -0.07 |] ~high:[| 0.6; 0.07 |] ~shape:[| 2 |])
      ~action_space:(Space.Discrete 3)
end
module S = Make (Owl.Neural.S.Algodiff.A)
module D = Make (Owl.Neural.D.Algodiff.A)
include S
