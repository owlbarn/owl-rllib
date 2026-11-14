(* Gymnasium environments as Ndenv values, run in a Python child process.

   [seed] seeds the first reset, as in env.reset(seed=seed); later resets
   continue from the environment's own generator, as Gymnasium does.
   Observations must be a Box; actions are Discrete ([discrete]) or a Box
   ([continuous]). Terminated and truncated come from Gymnasium unchanged. *)

open Owl_rllib

let script =
  {|
try:
    import gymnasium as gym
    import numpy as np
    env = gym.make(sys.argv[1], **({"render_mode": "rgb_array"} if sys.argv[3] == "1" else {}))
    seed = int(sys.argv[2])

    def space(s):
        if isinstance(s, gym.spaces.Discrete) and s.start == 0:
            return f'discrete {s.n}'
        if isinstance(s, gym.spaces.Box):
            return ' '.join(['box', str(len(s.shape)), *map(str, s.shape), hexes(s.low.ravel()), hexes(s.high.ravel())])
        raise ValueError(f'unsupported space {s}')

    reply(space(env.observation_space), space(env.action_space))
except Exception as e:
    fail(e)
    sys.exit(1)

def observation(obs):
    return hexes(np.asarray(obs, dtype=np.float64).ravel())

def handle(command, args):
    global seed
    if command == 'reset':
        obs, _ = env.reset(seed=seed)
        seed = None
        reply(observation(obs))
    elif command == 'render':
        frame = np.asarray(env.render(), dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Gym: expected an RGB frame')
        reply(frame.shape[1], frame.shape[0], frame.tobytes().hex())
    elif command == 'step':
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = int(args[0])
        else:
            action = np.array([float.fromhex(a) for a in args], dtype=env.action_space.dtype)
            action = action.reshape(env.action_space.shape)
        obs, reward, terminated, truncated, _ = env.step(action)
        reply(float(reward).hex(), int(terminated), int(truncated), observation(obs))

serve(handle)
|}

let rec split n xs = if n = 0 then [], xs else match xs with
  | x :: rest -> let first, rest = split (n - 1) rest in x :: first, rest
  | [] -> failwith "Gym: short reply"

let floats words = Array.of_list (List.map float_of_string words)

let space = function
  | "discrete" :: n :: rest -> Space.Discrete (int_of_string n), rest
  | "box" :: rank :: rest ->
      let shape, rest = split (int_of_string rank) rest in
      let shape = Array.of_list (List.map int_of_string shape) in
      let n = Array.fold_left ( * ) 1 shape in
      let low, rest = split n rest in
      let high, rest = split n rest in
      Space.box ~low:(floats low) ~high:(floats high) ~shape, rest
  | _ -> failwith "Gym: malformed space"

module Make (A : Owl_types_ndarray_algodiff.Sig) = struct
  let array shape words = A.of_array (Array.map A.float_to_elt (floats words)) shape

  let make ~render ~seed id ~encode =
    let process, info = Python.start ~name:("Gym " ^ id) ~script [| id; string_of_int seed; if render then "1" else "0" |] in
    let observation_space, rest = space info in
    let action_space, _ = space rest in
    let shape = match observation_space with
      | Space.Box b -> b.shape
      | Discrete _ -> invalid_arg "Gym: observations must be a Box" in
    let ended = ref true in
    let reset () =
      let obs = array shape (Python.request process "reset") in
      ended := false;
      obs
    in
    let step action =
      if !ended then invalid_arg "Ndenv.step: reset before stepping a new episode";
      match Python.request process ("step " ^ encode action) with
      | reward :: terminated :: truncated :: obs ->
          let terminated = terminated = "1" and truncated = truncated = "1" in
          ended := terminated || truncated;
          { Ndenv.obs = array shape obs; reward = float_of_string reward; terminated; truncated }
      | _ -> failwith ("Gym " ^ id ^ ": malformed step reply")
    in
    let frame () =
      match Python.request process "render" with
      | [ width; height; hex ] ->
          let width = int_of_string width and height = int_of_string height in
          let rgb = Bytes.init (String.length hex / 2) (fun i ->
            Char.chr (int_of_string ("0x" ^ String.sub hex (2 * i) 2))) in
          if Bytes.length rgb <> width * height * 3 then failwith "Gym: short RGB frame";
          width, height, rgb
      | _ -> failwith "Gym: malformed frame reply"
    in
    { Ndenv.observation_space; action_space; reset; step }, frame

  let discrete ~seed id : (A.arr, int) Ndenv.t =
    let env, _ = make ~render:false ~seed id ~encode:string_of_int in
    (match env.action_space with
     | Discrete _ -> ()
     | Box _ -> invalid_arg ("Gym.discrete: " ^ id ^ " has continuous actions"));
    env

  let discrete_with_renderer ~seed id =
    let env, frame = make ~render:true ~seed id ~encode:string_of_int in
    (match env.Ndenv.action_space with
     | Discrete _ -> ()
     | Box _ -> invalid_arg ("Gym.discrete_with_renderer: " ^ id ^ " has continuous actions"));
    env, frame

  let continuous ~seed id : (A.arr, A.arr) Ndenv.t =
    let encode action =
      let flat = A.reshape action [| A.numel action |] in
      String.concat " " (List.init (A.numel action) (fun i -> Python.hex (A.elt_to_float (A.get flat [| i |])))) in
    let env, _ = make ~render:false ~seed id ~encode in
    (match env.action_space with
     | Box _ -> ()
     | Discrete _ -> invalid_arg ("Gym.continuous: " ^ id ^ " has discrete actions"));
    env
end

module S = Make (Owl.Neural.S.Algodiff.A)
module D = Make (Owl.Neural.D.Algodiff.A)
