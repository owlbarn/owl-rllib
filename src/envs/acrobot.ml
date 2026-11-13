(* Gymnasium's default "book" dynamics, integrated with RK4, dt = 0.2. *)
module Make (A : Owl_types_ndarray_algodiff.Sig) = struct
  module E = Ndenv.Make (A)
  let clip x lo hi = Float.max lo (Float.min hi x)
  let rec wrap x =
    if x > Float.pi then wrap (x -. 2. *. Float.pi)
    else if x < -.Float.pi then wrap (x +. 2. *. Float.pi) else x
  let observe s = [| cos s.(0); sin s.(0); cos s.(1); sin s.(1); s.(2); s.(3) |]

  let derivative torque s =
    let t1, t2, v1, v2 = s.(0), s.(1), s.(2), s.(3) in
    let d1 = 3.5 +. cos t2 and d2 = 1.25 +. 0.5 *. cos t2 in
    let phi2 = 4.9 *. cos (t1 +. t2 -. Float.pi /. 2.) in
    let phi1 = -.0.5 *. v2 *. v2 *. sin t2 -. v2 *. v1 *. sin t2
               +. 14.7 *. cos (t1 -. Float.pi /. 2.) +. phi2 in
    let a2 = (torque +. d2 /. d1 *. phi1 -. 0.5 *. v1 *. v1 *. sin t2 -. phi2)
             /. (1.25 -. d2 *. d2 /. d1) in
    [| v1; v2; -.(d2 *. a2 +. phi1) /. d1; a2 |]

  let advance s action =
    if action < 0 || action > 2 then invalid_arg "Acrobot: action";
    let f = derivative (float_of_int (action - 1)) in
    let add scale k = Array.mapi (fun i x -> x +. scale *. k.(i)) s in
    let k1 = f s in
    let k2 = f (add 0.1 k1) in
    let k3 = f (add 0.1 k2) in
    let k4 = f (add 0.2 k3) in
    let s = Array.mapi (fun i x -> x +. 0.2 /. 6. *. (k1.(i) +. 2. *. k2.(i) +. 2. *. k3.(i) +. k4.(i))) s in
    s.(0) <- wrap s.(0); s.(1) <- wrap s.(1);
    s.(2) <- clip s.(2) (-4. *. Float.pi) (4. *. Float.pi);
    s.(3) <- clip s.(3) (-9. *. Float.pi) (9. *. Float.pi);
    let terminal = -.cos s.(0) -. cos (s.(0) +. s.(1)) > 1. in
    s, (if terminal then 0. else -1.), terminal

  let make ?(max_steps = 500) ?initial_state ~rng () =
    let reset_state () = match initial_state with
      | Some s -> if Array.length s <> 4 then invalid_arg "Acrobot: state" else Array.copy s
      | None -> Array.init 4 (fun _ ->
          (* Gymnasium casts the initial internal state to float32. *)
          Int32.float_of_bits (Int32.bits_of_float (Random.State.float rng 0.2 -. 0.1)))
    in
    let high = [| 1.; 1.; 1.; 1.; 4. *. Float.pi; 9. *. Float.pi |] in
    E.simulator ~max_steps ~reset_state ~observe ~advance
      ~observation_space:(Space.box ~low:(Array.map (~-.) high) ~high ~shape:[| 6 |])
      ~action_space:(Space.Discrete 3)
end
module S = Make (Owl.Neural.S.Algodiff.A)
module D = Make (Owl.Neural.D.Algodiff.A)
include S
