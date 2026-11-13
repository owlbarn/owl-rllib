(* A4. Linear approximation: tile-coded MountainCar and the deadly triad.
   OPENBLAS_NUM_THREADS=1 dune exec examples/classic/a4_approximation.exe *)
open Owl_rllib
module A = Owl.Dense.Ndarray.S

let features obs =
  let x = (A.get obs [| 0 |] +. 1.2) /. 1.8 *. 8.
  and v = (A.get obs [| 1 |] +. 0.07) /. 0.14 *. 8. in
  Array.init 8 (fun t ->
    let ix = min 8 (max 0 (int_of_float (floor (x +. float_of_int t /. 8.)))) in
    let iv = min 8 (max 0 (int_of_float (floor (v +. float_of_int ((3 * t) mod 8) /. 8.)))) in
    t * 81 + ix * 9 + iv)

(* Baird's seven-state counterexample, expected off-policy TD updates.
   Behavior's stationary distribution is uniform; target always goes to the
   lower state. Rewards are zero. Despite representable V*=0, weights diverge. *)
let baird () =
  let phi s = Array.init 8 (fun i ->
    (if s = 6 then (if i = 6 then 1. else if i = 7 then 2. else 0.)
     else if i = s then 2. else if i = 7 then 1. else 0.) /. sqrt 5.) in
  let weights = Array.init 8 (fun i -> if i = 6 then 10. else 1.) in
  let value s = Array.fold_left ( +. ) 0. (Array.map2 ( *. ) weights (phi s)) in
  for step = 0 to 10_000 do
    if step mod 2500 = 0 then
      Printf.printf "Baird expected TD: step %5d, weight norm %.3g\n" step
        (sqrt (Array.fold_left (fun sum x -> sum +. x *. x) 0. weights));
    let update = Array.make 8 0. in
    for s = 0 to 6 do
      let error = 0.99 *. value 6 -. value s in
      Array.iteri (fun i x -> update.(i) <- update.(i) +. error *. x /. 7.) (phi s)
    done;
    Array.iteri (fun i x -> weights.(i) <- x +. 0.01 *. update.(i)) weights
  done

let () =
  let off_policy = ref false in
  Arg.parse [ "--q-learning", Arg.Set off_policy, "Use semi-gradient Q-learning instead of SARSA" ] (fun _ -> ()) "A4";
  let rngs = Rng.create 42 in
  let env = Mountain_car.make ~rng:rngs.env () in
  let _, curve = Linear.S.train ~off_policy:!off_policy ~features ~n_features:(8 * 81)
      ~gamma:1. ~alpha:0.3 ~epsilon:0.01 ~episodes:500 ~rng:rngs.action env in
  for window = 0 to 4 do
    let xs = Array.sub curve (window * 100) 100 in
    Printf.printf "MountainCar episodes %3d-%3d: mean return %.1f\n"
      (window * 100 + 1) ((window + 1) * 100) (Array.fold_left ( +. ) 0. xs /. 100.)
  done;
  baird ()
