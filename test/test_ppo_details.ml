(* CleanRL's PPO details: normalisation wrappers, orthogonal initialisation,
   state-independent log_std, clipped value loss and learning-rate annealing. *)
open Owl_rllib
module A = Owl.Dense.Ndarray.D
module N = Neural.Make (Owl.Neural.D.Graph)
module P = Ppo.D
module AD = N.Algodiff

let lines path =
  let ic = open_in path in
  let rec go acc = match input_line ic with
    | line -> go (line :: acc)
    | exception End_of_file -> close_in ic; List.rev acc in
  go []

let hexes s = String.split_on_char ' ' (String.trim s) |> List.filter (( <> ) "") |> List.map float_of_string |> Array.of_list
let values x = A.to_array (A.reshape x [| A.numel x |])
let close tolerance = Alcotest.(array (float tolerance))

(* An environment that replays recorded raw transitions. *)
let replay ~shape ~action_space resets steps =
  let resets = ref resets and steps = ref steps in
  let pop r = match !r with x :: rest -> r := rest; x | [] -> failwith "replay: exhausted" in
  let n = Array.fold_left ( * ) 1 shape in
  { Ndenv.observation_space = Space.box ~low:(Array.make n neg_infinity) ~high:(Array.make n infinity) ~shape
  ; action_space
  ; reset = (fun () -> A.of_array (pop resets) shape)
  ; step = (fun _ -> let reward, terminated, truncated, obs = pop steps in
             { Ndenv.obs = A.of_array obs shape; reward; terminated; truncated }) }

let normalisation () =
  let groups = ref [] in
  List.iter (fun line -> match String.split_on_char ' ' line with
    | "E" :: _ -> groups := [] :: !groups
    | ("R" | "T") :: _ -> (match !groups with g :: rest -> groups := (line :: g) :: rest | [] -> assert false)
    | _ -> Alcotest.fail "malformed fixture") (lines "fixtures/ppo_normalise.txt");
  List.iter (fun group ->
    let group = List.rev group in
    let parse line = match String.split_on_char '|' (String.sub line 2 (String.length line - 2)) with
      | [ a; b ] -> `Reset (hexes a, hexes b)
      | [ a; b; c ] ->
          let a = hexes a in
          `Step (a.(0), a.(1) = 1., a.(2) = 1., Array.sub a 3 (Array.length a - 3), hexes b, (hexes c).(0))
      | _ -> failwith "malformed line" in
    let records = List.map parse group in
    let resets = List.filter_map (function `Reset (raw, _) -> Some raw | _ -> None) records in
    let steps = List.filter_map (function `Step (r, te, tr, raw, _, _) -> Some (r, te, tr, raw) | _ -> None) records in
    let dim = match records with `Reset (raw, _) :: _ -> Array.length raw | _ -> assert false in
    let env = replay ~shape:[| dim |] ~action_space:(Space.Discrete 1) resets steps
              |> Normalise.D.observation |> Normalise.reward ~gamma:0.99 in
    List.iter (function
      | `Reset (_, norm) -> Alcotest.check (close 1e-9) "reset observation" norm (values (env.reset ()))
      | `Step (_, _, _, _, norm, reward) ->
          let st = env.step 0 in
          Alcotest.check (close 1e-9) "observation" norm (values st.obs);
          Alcotest.(check (float 1e-9)) "reward" reward st.reward) records)
    !groups;
  let bounded = replay ~shape:[| 1 |] ~action_space:(Space.Discrete 1) [ [| 0. |] ] [ 100., false, false, [| 1e6 |] ]
                |> Normalise.D.observation ~bound:0.5 |> Normalise.reward ~bound:3. ~gamma:0.99 in
  ignore (bounded.reset ());
  let st = bounded.step 0 in
  Alcotest.(check (pair (float 0.) (float 0.))) "clipped" (0.5, 3.) ((values st.obs).(0), st.reward)

let orthogonal () =
  let check rows cols =
    let nn = N.mlp_with ~activation:N.Activation.Tanh ~init:(N.Orthogonal 0.5) ~seed:1 ~shape:[| rows |] ~hidden:[] ~outputs:cols in
    let w = AD.unpack_arr (N.Graph.mkpri nn).(2).(0) in
    let gram = if rows >= cols then A.dot (A.transpose w) w else A.dot w (A.transpose w) in
    let k = min rows cols in
    Alcotest.check (close 1e-12) (Printf.sprintf "%dx%d is orthogonal times the gain" rows cols)
      (values (A.mul_scalar (Owl.Mat.eye k) 0.25)) (values gram)
  in
  check 6 3; check 3 6; check 4 4

let state_independent_std () =
  let nn = N.gaussian_mlp ~activation:N.Activation.Tanh ~init:(N.Orthogonal 0.01) ~seed:2 ~shape:[| 3 |] ~hidden:[ 8 ] ~actions:2 in
  let x = A.gaussian [| 5; 3 |] in
  let output () = N.Graph.run (AD.Arr x) nn |> AD.unpack_arr in
  let log_std () = A.get_slice [ []; [ 2; 3 ] ] (output ()) in
  Alcotest.(check (array int)) "output [mean; log_std]" [| 5; 4 |] (A.shape (output ()));
  Alcotest.check (close 0.) "log_std starts at 0" (Array.make 10 0.) (values (log_std ()));
  Alcotest.(check bool) "small initial mean" true (A.(max' (abs (get_slice [ []; [ 0; 1 ] ] (output ())))) < 0.1);
  let stepper = N.Stepper.init (N.adam 0.1) (N.Graph.mkpri nn) in
  for _ = 1 to 3 do
    let y, _ = N.Graph.forward nn (AD.Arr x) in
    N.update nn stepper AD.Maths.(sum' (sqr (y - N.scalar 1.)))
  done;
  let s = log_std () in
  Alcotest.(check bool) "log_std learns" true (A.get s [| 0; 0 |] > 0.1);
  Alcotest.check (close 0.) "and stays the same for every state" (values (A.repeat (A.get_slice [ [ 0 ] ] s) [| 5; 1 |])) (values s)

let clipped_value () =
  let fixture = Hashtbl.create 5 in
  List.iter (fun line -> match String.index_opt line ' ' with
    | Some i -> Hashtbl.add fixture (String.sub line 0 i) (hexes (String.sub line i (String.length line - i)))
    | None -> ()) (lines "fixtures/ppo_value.txt");
  let column name = AD.Arr (A.of_array (Hashtbl.find fixture name) [| 5; 1 |]) in
  let error predicted = P.clipped_value_error ~clip:0.2 ~old:(column "old") ~predicted ~returns:(column "returns") in
  Alcotest.check (close 1e-12) "error" (Hashtbl.find fixture "error") (values (AD.unpack_arr (error (column "predicted"))));
  let gradient = AD.grad (fun v -> AD.Maths.sum' (error v)) (column "predicted") in
  Alcotest.check (close 1e-12) "gradient" (Hashtbl.find fixture "grad") (values (AD.unpack_arr gradient))

(* Annealing multiplies Adam's step: one step at scale 0.5 equals one step at
   half the learning rate; PPO's schedule runs 1, 1 - 1/U, ..., 1/U. *)
let annealing () =
  let module O = On_policy.Make (Owl.Neural.D.Graph) in
  let x = AD.Arr (A.gaussian [| 4; 3 |]) in
  let step ~scale ~lr =
    let nn = N.mlp ~seed:5 ~shape:[| 3 |] ~hidden:[ 6 ] ~outputs:2 in
    let y, _ = N.Graph.forward nn x in
    O.update_clipped ~step_scale:scale ~max_grad_norm:1e9 nn (N.Stepper.init (N.adam lr) (N.Graph.mkpri nn)) AD.Maths.(sum' (sqr y));
    List.concat_map (fun layer -> Array.to_list (Array.map (fun w -> values (AD.unpack_arr w)) layer)) (Array.to_list (N.Graph.mkpri nn))
    |> Array.concat in
  Alcotest.check (close 1e-15) "scale 0.5 = half the learning rate" (step ~scale:1. ~lr:0.005) (step ~scale:0.5 ~lr:0.01);
  Alcotest.(check bool) "scale 1 takes a step" true (step ~scale:1. ~lr:0.01 <> step ~scale:1e-9 ~lr:0.01)

(* The switches reach the training loop: a one-step bandit (reward = action)
   whose returns start far from the critic's values, so clipping binds. *)
let wiring () =
  let bandit () =
    { Ndenv.observation_space = Space.box ~low:[| 1. |] ~high:[| 1. |] ~shape:[| 1 |]
    ; action_space = Space.Discrete 2
    ; reset = (fun () -> A.of_array [| 1. |] [| 1 |])
    ; step = (fun a -> { Ndenv.obs = A.of_array [| 1. |] [| 1 |]; reward = 5. *. float_of_int a; terminated = true; truncated = false }) } in
  let train ~updates config =
    let rngs = Rng.create 0 in
    let policy = N.mlp ~seed:1 ~shape:[| 1 |] ~hidden:[ 4 ] ~outputs:2 and critic = N.mlp ~seed:2 ~shape:[| 1 |] ~hidden:[ 4 ] ~outputs:1 in
    ignore (P.train ~config:{ config with P.horizon = 4; batch_size = 4; learning_rate = 0.05 } ~updates ~rngs ~policy ~critic
              ~distribution:Policy.D.categorical (Vec_env.create [| bandit (); bandit () |]));
    List.map (fun nn -> Array.concat (List.concat_map (fun l -> Array.to_list (Array.map (fun w -> values (AD.unpack_arr w)) l))
                                        (Array.to_list (N.Graph.mkpri nn)))) [ policy; critic ] in
  let base = P.default in
  Alcotest.(check (list (array (float 0.)))) "first update runs at the full rate"
    (train ~updates:1 base) (train ~updates:1 { base with anneal_lr = true });
  List.iter2 (fun name (plain, annealed) -> Alcotest.(check bool) ("later updates of the " ^ name ^ " are annealed") true (plain <> annealed))
    [ "actor"; "critic" ] (List.combine (train ~updates:2 base) (train ~updates:2 { base with anneal_lr = true }));
  Alcotest.(check bool) "value clipping changes the critic" true
    (List.nth (train ~updates:1 base) 1 <> List.nth (train ~updates:1 { base with clip_value = true }) 1);
  Alcotest.(check (list (float 1e-15))) "schedule" [ 1.; 0.75; 0.5; 0.25 ]
    (List.map (P.learning_rate_fraction ~updates:4) [ 1; 2; 3; 4 ])

let () =
  Alcotest.run "PPO details"
    [ "ppo",
      [ Alcotest.test_case "normalisation matches Gymnasium" `Quick normalisation
      ; Alcotest.test_case "orthogonal initialisation" `Quick orthogonal
      ; Alcotest.test_case "state-independent log_std" `Quick state_independent_std
      ; Alcotest.test_case "clipped value loss" `Quick clipped_value
      ; Alcotest.test_case "learning-rate annealing" `Quick annealing
      ; Alcotest.test_case "switches reach training" `Quick wiring ] ]
