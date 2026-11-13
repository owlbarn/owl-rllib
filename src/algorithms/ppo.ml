module Make (Graph : Owl_neural_graph_sig.Sig) = struct
  include On_policy.Make (Graph)
  open Algodiff
  type config =
    { horizon : int; epochs : int; batch_size : int; gamma : float; lambda : float
    ; learning_rate : float; clip : float; entropy_coef : float; value_coef : float
    ; max_grad_norm : float; normalise_advantage : bool
    ; clip_value : bool (* clip the value update like the ratio, as CleanRL *)
    ; anneal_lr : bool (* learning rate falls linearly to 0 over the updates *) }
  let default =
    { horizon = 128; epochs = 4; batch_size = 64; gamma = 0.99; lambda = 0.95
    ; learning_rate = 3e-4; clip = 0.2; entropy_coef = 0.; value_coef = 0.5
    ; max_grad_norm = 0.5; normalise_advantage = true; clip_value = false; anneal_lr = false }

  let surrogate ~clip ~log_prob ~old_log_prob ~advantages =
    let ratio = Maths.exp Maths.(log_prob - old_log_prob) in
    let clipped = Maths.(min2 (scalar (1. +. clip)) (max2 (scalar (1. -. clip)) ratio)) in
    Maths.(neg (mean (min2 (ratio * advantages) (clipped * advantages))))

  (* Per-sample squared error of a value prediction kept within [clip] of the
     rollout's value: the larger of the clipped and unclipped errors. *)
  let clipped_value_error ~clip ~old ~predicted ~returns =
    let clipped = Maths.(old + min2 (scalar clip) (max2 (scalar (-. clip)) (predicted - old))) in
    Maths.(max2 (sqr (predicted - returns)) (sqr (clipped - returns)))

  (* Linear annealing: update u of [updates] runs at this fraction of the
     learning rate, from 1 down to 1 / updates. *)
  let learning_rate_fraction ~updates update = 1. -. float_of_int (update - 1) /. float_of_int updates

  let train ?(config = default) ~updates ~rngs ~policy ~critic ~distribution env =
    if updates <= 0 || config.horizon <= 0 || config.epochs <= 0 || config.batch_size <= 0
       || config.clip < 0. || config.clip >= 1. || config.max_grad_norm <= 0.
    then invalid_arg "Ppo.train: invalid configuration";
    let actor_stepper = Stepper.init (adam config.learning_rate) (Graph.mkpri policy) in
    let critic_stepper = Stepper.init (adam config.learning_rate) (Graph.mkpri critic) in
    let state = start env in
    for update = 1 to updates do
      let step_scale = if config.anneal_lr then learning_rate_fraction ~updates update else 1. in
      let rollout = collect ~horizon:config.horizon ~gamma:config.gamma ~lambda:config.lambda
          ~rng:rngs.Rng.action ~policy ~critic ~distribution env state in
      let n = Array.length rollout.R.data in
      for _ = 1 to config.epochs do
        let order = Rollout.permutation ~rng:rngs.sample n in
        let offset = ref 0 in
        while !offset < n do
          let count = min config.batch_size (n - !offset) in
          let indices = Array.sub order !offset count in
          let obs, actions, old_log_prob, advantages, returns = select rollout indices in
          let advantages = if config.normalise_advantage && count > 1 then normalise advantages else advantages in
          let output, _ = Graph.forward policy obs in
          let log_prob = distribution.Policy.log_prob output actions in
          let pg = surrogate ~clip:config.clip ~log_prob ~old_log_prob ~advantages:(column advantages) in
          let entropy = distribution.entropy output in
          update_clipped ~step_scale ~max_grad_norm:config.max_grad_norm policy actor_stepper
            Maths.(pg - scalar config.entropy_coef * mean entropy);
          let predicted, _ = Graph.forward critic obs in
          let error = if config.clip_value then clipped_value_error ~clip:config.clip ~old:(old_values rollout indices) ~predicted ~returns
            else Maths.sqr Maths.(predicted - returns) in
          update_clipped ~step_scale ~max_grad_norm:config.max_grad_norm critic critic_stepper
            Maths.(scalar config.value_coef * mean error);
          offset := !offset + count
        done
      done
    done;
    Array.of_list (List.rev state.completed)
end
module S = Make (Owl.Neural.S.Graph)
module D = Make (Owl.Neural.D.Graph)
