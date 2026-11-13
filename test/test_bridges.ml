open Owl_rllib
open Owl_rllib_bridges
module A = Owl.Dense.Ndarray.S
module N = Neural.Make (Owl.Neural.S.Graph)

let values x = A.to_array (A.reshape x [| A.numel x |])
let floats = Alcotest.(array (float 0.))

(* An episode of three steps; observation k is [k; 10 + k] in a [1; 1; 2] box. *)
let counter () =
  let t = ref 0 in
  let frame () = A.of_array [| float_of_int !t; 10. +. float_of_int !t |] [| 1; 1; 2 |] in
  { Ndenv.observation_space = Space.box ~low:[| 0.; -1. |] ~high:[| 1.; 2. |] ~shape:[| 1; 1; 2 |]
  ; action_space = Space.Discrete 1
  ; reset = (fun () -> t := 0; frame ())
  ; step = (fun _ -> incr t; { Ndenv.obs = frame (); reward = 0.; terminated = !t = 3; truncated = false }) }

let frame_stack () =
  let env = Frame_stack.S.wrap ~frames:3 (counter ()) in
  (match env.observation_space with
   | Space.Box b ->
       Alcotest.(check (array int)) "shape" [| 1; 1; 6 |] b.shape;
       Alcotest.check floats "low" [| 0.; -1.; 0.; -1.; 0.; -1. |] b.low;
       Alcotest.check floats "high" [| 1.; 2.; 1.; 2.; 1.; 2. |] b.high
   | Discrete _ -> Alcotest.fail "box expected");
  Alcotest.check floats "reset repeats the first frame" [| 0.; 10.; 0.; 10.; 0.; 10. |] (values (env.reset ()));
  Alcotest.check floats "newest last" [| 0.; 10.; 0.; 10.; 1.; 11. |] (values (env.step 0).obs);
  Alcotest.check floats "oldest dropped" [| 0.; 10.; 1.; 11.; 2.; 12. |] (values (env.step 0).obs);
  let last = env.step 0 in
  Alcotest.(check bool) "terminated passes through" true last.terminated;
  Alcotest.check floats "final observation" [| 1.; 11.; 2.; 12.; 3.; 13. |] (values last.obs);
  Alcotest.check floats "reset clears the stack" [| 0.; 10.; 0.; 10.; 0.; 10. |] (values (env.reset ()));
  let single = Frame_stack.S.wrap ~frames:1 (counter ()) in
  ignore (single.reset ());
  Alcotest.check floats "one frame is the observation" [| 1.; 11. |] (values (single.step 0).obs)

let cnn () =
  let make () = N.cnn ~seed:3 ~shape:[| 12; 16; 2 |] ~convs:[ 4, 2, 3; 3, 1, 4 ] ~hidden:[ 8 ] ~outputs:3 in
  let nn = make () in
  let count = Array.fold_left (Array.fold_left (fun n w -> n + A.numel (N.Algodiff.unpack_arr w))) 0 (N.Graph.mkpri nn) in
  (* 12x16 -> 5x7x3 -> 3x5x4 = 60 features *)
  Alcotest.(check int) "parameters" ((4 * 4 * 2 * 3 + 3) + (3 * 3 * 3 * 4 + 4) + (60 * 8 + 8) + (8 * 3 + 3)) count;
  let x = A.uniform [| 5; 12; 16; 2 |] in
  let y = N.Graph.run (N.Algodiff.Arr x) nn |> N.Algodiff.unpack_arr in
  Alcotest.(check (array int)) "output" [| 5; 3 |] (A.shape y);
  let y' = N.Graph.run (N.Algodiff.Arr x) (make ()) |> N.Algodiff.unpack_arr in
  Alcotest.(check bool) "same seed, same network" true (A.equal y y')

(* Bridge tests compare with test/reference/bridges.py's output and are skipped
   when the Python packages are missing. *)
let requires modules f () =
  let python = Option.value (Sys.getenv_opt "OWL_RLLIB_PYTHON") ~default:"python3" in
  if Sys.command (Printf.sprintf "%s -I -c 'import %s' 2>/dev/null" (Filename.quote python) modules) = 0 then f ()
  else (Printf.printf "Python lacks %s\n" modules; Alcotest.skip ())

let lines path =
  let ic = open_in path in
  let rec go acc = match input_line ic with
    | line -> go (String.split_on_char ' ' line :: acc)
    | exception End_of_file -> close_in ic; List.rev acc in
  go []

let hexes words = Array.of_list (List.map float_of_string words)
let raises_failure name f = match f () with
  | _ -> Alcotest.fail (name ^ ": no failure")
  | exception Failure _ -> ()

let gym_fixture () =
  let reset = ref (fun () -> A.zeros [| 1 |]) and step = ref (fun _ -> assert false) in
  List.iter (function
    | [ "E"; "CartPole-v1"; seed ] ->
        let env = Gym.S.discrete ~seed:(int_of_string seed) "CartPole-v1" in
        Alcotest.(check bool) "CartPole spaces" true
          (env.action_space = Space.Discrete 2
           && (match env.observation_space with Space.Box b -> b.shape = [| 4 |] && b.high.(1) = 0x1.fffffep+127 (* float32 max *) | _ -> false));
        reset := env.reset;
        step := fun words -> env.step (int_of_string (List.hd words))
    | [ "E"; "Pendulum-v1"; seed ] ->
        let env = Gym.S.continuous ~seed:(int_of_string seed) "Pendulum-v1" in
        Alcotest.(check bool) "Pendulum actions" true
          (env.action_space = Space.box ~low:[| -2. |] ~high:[| 2. |] ~shape:[| 1 |]);
        reset := env.reset;
        step := fun words -> env.step (A.of_array (hexes [ List.hd words ]) [| 1 |])
    | "R" :: obs -> Alcotest.check floats "reset" (hexes obs) (values (!reset ()))
    | "T" :: action :: reward :: terminated :: truncated :: obs ->
        let st = !step [ action ] in
        Alcotest.check floats "step" (hexes obs) (values st.obs);
        Alcotest.(check (float 0.)) "reward" (float_of_string reward) st.reward;
        Alcotest.(check (pair bool bool)) "flags" (terminated = "1", truncated = "1") (st.terminated, st.truncated)
    | _ -> Alcotest.fail "malformed fixture")
    (lines "fixtures/bridge_gym.txt")

let gym_contract () =
  let env = Gym.S.discrete ~seed:0 "CartPole-v1" in
  Alcotest.check_raises "step before reset" (Invalid_argument "Ndenv.step: reset before stepping a new episode")
    (fun () -> ignore (env.step 0));
  ignore (env.reset ());
  raises_failure "invalid action" (fun () -> env.step 5);
  let rec finish () = if not (env.step 1).terminated then finish () in
  finish ();
  Alcotest.check_raises "step after the end" (Invalid_argument "Ndenv.step: reset before stepping a new episode")
    (fun () -> ignore (env.step 0));
  raises_failure "unknown id" (fun () -> Gym.S.discrete ~seed:0 "NoSuchEnv-v0");
  Alcotest.check_raises "wrong action space" (Invalid_argument "Gym.continuous: CartPole-v1 has discrete actions")
    (fun () -> ignore (Gym.S.continuous ~seed:0 "CartPole-v1"))

let doom_fixture () =
  let env = Doom.S.make ~seed:7 "basic" in
  Alcotest.(check bool) "spaces" true
    (env.action_space = Space.Discrete 3
     && (match env.observation_space with Space.Box b -> b.shape = [| 60; 80; 1 |] | _ -> false));
  let sums obs = Array.map (fun v -> int_of_float (Float.round (v *. 1020.))) (values obs) in
  let total obs = Array.fold_left ( + ) 0 (sums obs) in
  let ends = ref [] in
  List.iteri (fun i -> function
    | [ "R"; sum ] -> Alcotest.(check int) "reset total" (int_of_string sum) (total (env.reset ()))
    | "R" :: frame when i = 0 ->
        Alcotest.(check (array int)) "first frame" (Array.of_list (List.map int_of_string frame)) (sums (env.reset ()))
    | [ "T"; action; reward; terminated; truncated; sum ] ->
        let st = env.step (int_of_string action) in
        Alcotest.(check int) "frame total" (int_of_string sum) (total st.obs);
        Alcotest.(check (float 0.)) "reward" (float_of_string reward) st.reward;
        Alcotest.(check (pair bool bool)) "flags" (terminated = "1", truncated = "1") (st.terminated, st.truncated);
        if st.terminated || st.truncated then ends := (st.terminated, st.truncated) :: !ends
    | _ -> Alcotest.fail "malformed fixture")
    (lines "fixtures/bridge_doom.txt");
  Alcotest.(check bool) "fixture covers a kill and a time limit" true
    (List.mem (true, false) !ends && List.mem (false, true) !ends)

let () =
  Alcotest.run "bridges"
    [ "wrappers", [ Alcotest.test_case "frame stack" `Quick frame_stack; Alcotest.test_case "cnn" `Quick cnn ]
    ; "bridges",
      [ Alcotest.test_case "gymnasium trajectories" `Quick (requires "gymnasium" gym_fixture)
      ; Alcotest.test_case "gymnasium contract" `Quick (requires "gymnasium" gym_contract)
      ; Alcotest.test_case "vizdoom trajectories" `Quick (requires "vizdoom" doom_fixture) ] ]
