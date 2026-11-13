(* Vanilla DQN. Turning the target network off is an ablation
   (examples/classic/a5_dqn.ml). *)
module Make (Graph : Owl_neural_graph_sig.Sig) = struct
  include Neural.Make (Graph)
  open Algodiff

  type config =
    { gamma : float; learning_rate : float; capacity : int; batch_size : int
    ; learning_starts : int; train_every : int; target_every : int
    ; epsilon_start : float; epsilon_end : float; exploration_steps : int
    ; target_network : bool }

  let default =
    { gamma = 0.99; learning_rate = 0.001; capacity = 10_000; batch_size = 64
    ; learning_starts = 1000; train_every = 4; target_every = 500
    ; epsilon_start = 1.; epsilon_end = 0.05; exploration_steps = 10_000
    ; target_network = true }

  let epsilon config step =
    let fraction = min 1. (float_of_int step /. float_of_int config.exploration_steps) in
    config.epsilon_start +. fraction *. (config.epsilon_end -. config.epsilon_start)

  let train ?(config = default) ~steps ~rngs ~network (env : int E.t) =
    if steps <= 0 || config.batch_size <= 0 || config.train_every <= 0
       || config.target_every <= 0 || config.exploration_steps <= 0
       || config.learning_starts < 0
    then invalid_arg "Dqn.train: invalid counts";
    let n = n_actions env in
    let replay = B.create config.capacity in
    let target = Graph.copy network in
    let optimiser = Stepper.init (adam config.learning_rate) (Graph.mkpri network) in
    let obs = ref (env.reset ()) and total = ref 0. and curve = ref [] in
    for step = 1 to steps do
      let action =
        if Random.State.float rngs.Rng.action 1. < epsilon config (step - 1)
        then Random.State.int rngs.action n
        else predict network !obs |> Dp.argmax
      in
      let st = env.step action in
      B.add replay { obs = !obs; action; reward = st.reward; next_obs = st.obs
                   ; terminated = st.terminated; truncated = st.truncated };
      total := !total +. st.reward;
      obs := st.obs;
      if st.terminated || st.truncated then (
        curve := !total :: !curve; total := 0.; obs := env.reset ());
      if step >= config.learning_starts && step mod config.train_every = 0 then (
        let samples = B.sample ~rng:rngs.sample ~batch_size:config.batch_size replay in
        (* Compute fixed TD targets before tagging the online weights. *)
        let next = batch (Array.map (fun (tr : B.transition) -> tr.next_obs) samples) in
        let bootstrap_net = if config.target_network then target else network in
        let q_next = Graph.run next bootstrap_net |> unpack_arr in
        let targets = Array.mapi (fun i (tr : B.transition) ->
          let max_q = ref neg_infinity in
          for a = 0 to n - 1 do
            max_q := Float.max !max_q (A.get q_next [| i; a |] |> A.elt_to_float)
          done;
          Buffer.td_target ~gamma:config.gamma ~reward:tr.reward ~terminated:tr.terminated ~next_value:!max_q) samples |> column in
        let inputs = batch (Array.map (fun (tr : B.transition) -> tr.obs) samples) in
        let actions = Array.map (fun (tr : B.transition) -> tr.action) samples |> one_hot n in
        let q, _ = Graph.forward network inputs in
        let selected = Maths.sum ~axis:1 ~keep_dims:true Maths.(q * actions) in
        update network optimiser Maths.(mean (sqr (selected - targets))));
      if config.target_network && step mod config.target_every = 0 then
        Graph.update target (Array.map (Array.map (fun w -> Arr (A.copy (unpack_arr w)))) (Graph.mkpri network))
    done;
    Array.of_list (List.rev !curve)
end
module S = Make (Owl.Neural.S.Graph)
module D = Make (Owl.Neural.D.Graph)
