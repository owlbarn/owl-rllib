(* A2. Dynamic programming: solving an MDP when the model is known.

   Policy evaluation computes the value of a given policy. Policy iteration
   and value iteration find the best policy.

   Run: dune exec examples/classic/a2_dp.exe *)

open Owl_rllib

let () =
  (* 1. Policy evaluation. Turn the Bellman equation into an update rule and
     sweep over the states until the values stop changing. *)
  let grid = Gridworld.sutton_4x4 () in
  let uniform = Array.make_matrix grid.mdp.n_states 4 0.25 in
  print_endline "Random policy on the 4x4 grid (Sutton & Barto, figure 4.1):";
  Gridworld.print_values grid (Dp.policy_evaluation ~gamma:1. grid.mdp uniform);

  (* 2. Policy improvement. Acting greedily with respect to those values is
     already better than acting at random. *)
  let v = Dp.policy_evaluation ~gamma:1. grid.mdp uniform in
  print_endline "\nGreedy policy with respect to the values above:";
  Gridworld.print_policy grid (Dp.greedy (Dp.q_from_v ~gamma:1. grid.mdp v));

  (* 3. Policy iteration alternates evaluation and improvement until the
     policy no longer changes. Value iteration folds both into one update:
       v(s) <- max_a sum_s' p(s'|s,a) [r + gamma v(s')] *)
  let maze = Gridworld.make [| "S...#..."; ".##.#.#."; ".#....#."; ".#.##.#G" |] in
  let gamma = 0.95 in
  let pi_policy, pi_v = Dp.policy_iteration ~gamma maze.mdp in
  let vi_policy, vi_v = Dp.value_iteration ~gamma maze.mdp in
  print_endline "\nA maze, optimal values:";
  Gridworld.print_values maze vi_v;
  print_endline "\nOptimal policy:";
  Gridworld.print_policy maze vi_policy;
  let diff = ref 0. in
  Array.iteri (fun s x -> diff := Float.max !diff (Float.abs (x -. vi_v.(s)))) pi_v;
  Printf.printf
    "\nPolicy iteration and value iteration: largest difference in values %.2e, same policy: %b\n"
    !diff
    (pi_policy = vi_policy);

  (* 4. A stochastic world. On the frozen lake the agent slips sideways two
     times out of three. With gamma = 1 and reward 1 at the goal, the value of
     the start state is the probability of ever reaching the goal. *)
  let lake = Frozen_lake.make () in
  let _, v = Dp.value_iteration ~gamma:1. lake in
  let v_random = Dp.policy_evaluation ~gamma:1. lake (Array.make_matrix 16 4 0.25) in
  Printf.printf
    "\nFrozen lake, probability of reaching the goal from the start:\n  random policy %.3f, optimal policy %.3f\n"
    v_random.(0)
    v.(0)
