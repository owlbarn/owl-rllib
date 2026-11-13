(* A1. Markov decision processes.

   An MDP is a set of states, a set of actions, and for every (state, action)
   pair a distribution over (next state, reward). This file builds one, looks
   inside it, samples episodes from it, and checks the Bellman equation.

   Run: dune exec examples/classic/a1_mdp.exe *)

open Owl_rllib

let () =
  (* The 4x4 grid of Sutton & Barto, example 4.1. The two corners marked G are
     terminal. Every move costs 1, so a good policy reaches a corner quickly. *)
  let grid = Gridworld.sutton_4x4 () in
  let mdp = grid.mdp in
  Printf.printf "states: %d, actions: %d (0 up, 1 right, 2 down, 3 left)\n\n" mdp.n_states mdp.n_actions;

  (* 1. The model: what happens in state 1 for each action. *)
  print_endline "Transitions from state 1:";
  Array.iteri
    (fun a outcomes ->
      List.iter
        (fun (tr : Mdp.transition) ->
          Printf.printf
            "  action %d -> state %2d, reward %4.1f, prob %.2f%s\n"
            a tr.next tr.reward tr.prob
            (if tr.terminal then "  (episode ends)" else ""))
        outcomes)
    mdp.transitions.(1);

  (* 2. A policy maps states to actions. Here: pick an action at random. *)
  let rng = Random.State.make [| 0 |] in
  let random_policy _ = Random.State.int rng 4 in

  (* 3. An episode is a sequence of states, actions and rewards. Its return is
     the discounted sum of rewards: G = r1 + gamma r2 + gamma^2 r3 + ... *)
  let gamma = 1. in
  let episode start =
    let rec go s discount g =
      let tr = Mdp.sample_transition rng mdp s (random_policy s) in
      let g = g +. (discount *. tr.reward) in
      if tr.terminal then g else go tr.next (discount *. gamma) g
    in
    go start 1. 0.
  in
  Printf.printf "\nReturns of five random episodes from state 1:";
  for _ = 1 to 5 do
    Printf.printf " %.0f" (episode 1)
  done;

  (* 4. The value of a state under a policy is its expected return. The
     simplest estimate is the average over many episodes. *)
  let n = 20_000 in
  let estimate s = List.init n (fun _ -> episode s) |> List.fold_left ( +. ) 0. |> fun x -> x /. float_of_int n in
  Printf.printf "\n\nAverage return over %d episodes, per start state:\n" n;
  let sampled = Array.init mdp.n_states (fun s -> if grid.cells.(s / 4).[s mod 4] = 'G' then 0. else estimate s) in
  Gridworld.print_values grid sampled;

  (* 5. With the model in hand we do not need to sample. The values satisfy
     the Bellman equation
       v(s) = sum_a pi(a|s) sum_s' p(s'|s,a) [r + gamma v(s')]
     and can be computed exactly (next chapter shows how). *)
  let uniform = Array.make_matrix mdp.n_states 4 0.25 in
  let v = Dp.policy_evaluation ~gamma mdp uniform in
  print_endline "\nExact values of the random policy:";
  Gridworld.print_values grid v;

  (* Check the Bellman equation at one state by hand. *)
  let s = 1 in
  let rhs = Array.init 4 (Dp.backup ~gamma mdp v s) |> Array.fold_left (fun acc q -> acc +. (0.25 *. q)) 0. in
  Printf.printf "\nBellman equation at state %d: v = %.4f, right-hand side = %.4f\n" s v.(s) rhs
