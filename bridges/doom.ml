(* VizDoom scenarios as Ndenv values, run in a Python child process.

   Action i presses button i alone: in basic, defend_the_center and
   health_gathering that means move/turn left, move/turn right, and attack or
   move forward. One step repeats the action for [frame_skip] tics and returns
   their summed reward. The screen is rendered at 160x120 in grayscale, then
   averaged over [downsample] x [downsample] blocks and scaled to [0, 1]; the
   observation has shape [height; width; 1], ready for Frame_stack.

   An episode that reaches the scenario's time limit is truncated; one that
   ends any other way (kill, death) is terminated. VizDoom renders no screen
   after the episode has ended, so the final observation repeats the last
   rendered one. *)

open Owl_rllib

let script =
  {|
try:
    import vizdoom as vzd
    scenario, seed, frame_skip = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, scenario + '.cfg'))
    game.set_window_visible(False)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_seed(seed)
    game.init()
    buttons = len(game.get_available_buttons())
    reply(buttons, game.get_screen_height(), game.get_screen_width())
except Exception as e:
    fail(e)
    sys.exit(1)

timeout = game.get_episode_timeout()
limit = game.get_episode_start_time() + timeout
last = None

def screen():
    global last
    state = game.get_state()
    if state is not None:
        last = state.screen_buffer.tobytes().hex()
    return last

def handle(command, args):
    if command == 'reset':
        game.new_episode()
        reply(screen())
    elif command == 'step':
        action = [0] * buttons
        action[int(args[0])] = 1
        reward = game.make_action(action, frame_skip)
        ended = game.is_episode_finished()
        truncated = ended and timeout > 0 and game.get_episode_time() >= limit
        reply(float(reward).hex(), int(ended and not truncated), int(truncated), screen())

serve(handle)
|}

let hex_digit c = match c with
  | '0' .. '9' -> Char.code c - 48
  | 'a' .. 'f' -> Char.code c - 87
  | _ -> failwith "Doom: malformed screen"

module Make (A : Owl_types_ndarray_algodiff.Sig) = struct
  let make ?(frame_skip = 4) ?(downsample = 2) ~seed scenario : (A.arr, int) Ndenv.t =
    if frame_skip <= 0 || downsample <= 0 then invalid_arg "Doom.make: positive frame_skip and downsample required";
    let name = "VizDoom " ^ scenario in
    let process, info = Python.start ~name ~script [| scenario; string_of_int seed; string_of_int frame_skip |] in
    let buttons, rows, columns = match List.map int_of_string info with
      | [ b; r; c ] -> b, r, c
      | _ -> failwith (name ^ ": malformed reply") in
    if rows mod downsample <> 0 || columns mod downsample <> 0 then invalid_arg "Doom.make: downsample must divide 120 and 160";
    let height = rows / downsample and width = columns / downsample in
    let scale = 1. /. float_of_int (downsample * downsample * 255) in
    let screen hex =
      if String.length hex <> 2 * rows * columns then failwith (name ^ ": malformed screen");
      let pixel r c = let i = 2 * (r * columns + c) in 16 * hex_digit hex.[i] + hex_digit hex.[i + 1] in
      A.init [| height; width; 1 |] (fun i ->
        let r = i / width * downsample and c = i mod width * downsample in
        let total = ref 0 in
        for dr = 0 to downsample - 1 do
          for dc = 0 to downsample - 1 do total := !total + pixel (r + dr) (c + dc) done
        done;
        A.float_to_elt (float_of_int !total *. scale))
    in
    let ended = ref true in
    let reset () = match Python.request process "reset" with
      | [ hex ] -> ended := false; screen hex
      | _ -> failwith (name ^ ": malformed reset reply") in
    let step action =
      if !ended then invalid_arg "Ndenv.step: reset before stepping a new episode";
      if action < 0 || action >= buttons then invalid_arg "Doom: action";
      match Python.request process ("step " ^ string_of_int action) with
      | [ reward; terminated; truncated; hex ] ->
          let terminated = terminated = "1" and truncated = truncated = "1" in
          ended := terminated || truncated;
          { Ndenv.obs = screen hex; reward = float_of_string reward; terminated; truncated }
      | _ -> failwith (name ^ ": malformed step reply")
    in
    let n = height * width in
    { Ndenv.observation_space = Space.box ~low:(Array.make n 0.) ~high:(Array.make n 1.) ~shape:[| height; width; 1 |]
    ; action_space = Space.Discrete buttons; reset; step }
end

module S = Make (Owl.Neural.S.Algodiff.A)
module D = Make (Owl.Neural.D.Algodiff.A)
