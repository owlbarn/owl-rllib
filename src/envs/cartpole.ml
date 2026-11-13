(* CartPole-v1, explicit Euler integration; see Gymnasium classic_control. *)
module Make (A : Owl_types_ndarray_algodiff.Sig) = struct
  module E = Ndenv.Make (A)

  let advance s action =
    if action < 0 || action > 1 then invalid_arg "Cartpole: action";
    let x, v, theta, omega = s.(0), s.(1), s.(2), s.(3) in
    let force = if action = 1 then 10. else -10. in
    let c = cos theta and sn = sin theta in
    let temp = (force +. 0.05 *. omega *. omega *. sn) /. 1.1 in
    let angular = (9.8 *. sn -. c *. temp) /. (0.5 *. (4. /. 3. -. 0.1 *. c *. c /. 1.1)) in
    let accel = temp -. 0.05 *. angular *. c /. 1.1 in
    let s = [| x +. 0.02 *. v; v +. 0.02 *. accel;
               theta +. 0.02 *. omega; omega +. 0.02 *. angular |] in
    let terminal = abs_float s.(0) > 2.4 || abs_float s.(2) > 12. *. Float.pi /. 180. in
    s, 1., terminal

  let make ?(max_steps = 500) ?initial_state ~rng () =
    let reset_state () = match initial_state with
      | Some s -> if Array.length s <> 4 then invalid_arg "Cartpole: state" else Array.copy s
      | None -> Array.init 4 (fun _ -> Random.State.float rng 0.1 -. 0.05)
    in
    let theta = 24. *. Float.pi /. 180. in
    E.simulator ~max_steps ~reset_state ~observe:Fun.id ~advance
      ~observation_space:(Space.box ~low:[| -4.8; neg_infinity; -.theta; neg_infinity |]
                           ~high:[| 4.8; infinity; theta; infinity |] ~shape:[| 4 |])
      ~action_space:(Space.Discrete 2)
end
module S = Make (Owl.Neural.S.Algodiff.A)
module D = Make (Owl.Neural.D.Algodiff.A)
include S
