module Make (A : Owl_types_ndarray_algodiff.Sig) = struct
  module E = Ndenv.Make (A)
  let clip x lo hi = Float.max lo (Float.min hi x)
  let angle x = x -. 2. *. Float.pi *. floor ((x +. Float.pi) /. (2. *. Float.pi))
  let observe s = [| cos s.(0); sin s.(0); s.(1) |]

  let advance s action =
    if A.shape action <> [| 1 |] then invalid_arg "Pendulum: action shape";
    let u = clip (A.get action [| 0 |] |> A.elt_to_float) (-2.) 2. in
    let cost = angle s.(0) ** 2. +. 0.1 *. s.(1) ** 2. +. 0.001 *. u ** 2. in
    let v = clip (s.(1) +. (15. *. sin s.(0) +. 3. *. u) *. 0.05) (-8.) 8. in
    [| s.(0) +. v *. 0.05; v |], -.cost, false

  let make ?(max_steps = 200) ?initial_state ~rng () =
    let reset_state () = match initial_state with
      | Some s -> if Array.length s <> 2 then invalid_arg "Pendulum: state" else Array.copy s
      | None ->
        let theta = Random.State.float rng (2. *. Float.pi) -. Float.pi in
        let velocity = Random.State.float rng 2. -. 1. in
        [| theta; velocity |]
    in
    E.simulator ~max_steps ~reset_state ~observe ~advance
      ~observation_space:(Space.box ~low:[| -1.; -1.; -8. |] ~high:[| 1.; 1.; 8. |] ~shape:[| 3 |])
      ~action_space:(Space.box ~low:[| -2. |] ~high:[| 2. |] ~shape:[| 1 |])
end
module S = Make (Owl.Neural.S.Algodiff.A)
module D = Make (Owl.Neural.D.Algodiff.A)
include S
