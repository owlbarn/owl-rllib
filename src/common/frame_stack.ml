(* Stack the last [frames] observations along their last axis (the channel
   axis of NHWC images), so a policy that sees one observation sees motion.
   Oldest first. A reset fills the stack with copies of the first frame; the
   final observation of an episode still ends with its true final frame. *)
module Make (A : Owl_types_ndarray_algodiff.Sig) = struct
  let wrap ~frames (env : (A.arr, 'action) Ndenv.t) : (A.arr, 'action) Ndenv.t =
    if frames <= 0 then invalid_arg "Frame_stack.wrap: positive frames required";
    let low, high, shape = match env.observation_space with
      | Space.Box b -> b.low, b.high, b.shape
      | Discrete _ -> invalid_arg "Frame_stack.wrap: Box observations required" in
    let axis = Array.length shape - 1 in
    let channels = shape.(axis) in
    (* row-major bounds: each block of [channels] values repeats [frames] times *)
    let tile xs = Array.init (Array.length xs * frames) (fun i -> xs.(i / (channels * frames) * channels + i mod channels)) in
    let observation_space =
      Space.box ~low:(tile low) ~high:(tile high) ~shape:(Array.mapi (fun i d -> if i = axis then d * frames else d) shape) in
    let stack = ref [||] in
    let observe () = A.concatenate ~axis !stack in
    let reset () =
      stack := Array.make frames (env.reset ());
      observe ()
    in
    let step action =
      let st = env.step action in
      stack := Array.append (Array.sub !stack 1 (frames - 1)) [| st.Ndenv.obs |];
      { st with obs = observe () }
    in
    { env with observation_space; reset; step }
end

module S = Make (Owl.Neural.S.Algodiff.A)
module D = Make (Owl.Neural.D.Algodiff.A)
