(* Synchronous advantage actor-critic: one update per fresh rollout, no replay. *)
module Make (Graph : Owl_neural_graph_sig.Sig) = struct
  include On_policy.Make (Graph)
  open Algodiff
  type config =
    { horizon : int; gamma : float; lambda : float; learning_rate : float
    ; entropy_coef : float; value_coef : float; max_grad_norm : float }
  let default =
    { horizon = 5; gamma = 0.99; lambda = 1.; learning_rate = 7e-4
    ; entropy_coef = 0.01; value_coef = 0.5; max_grad_norm = 0.5 }

  let actor_loss ~log_prob ~advantages ~entropy ~entropy_coef =
    Maths.(neg (mean (log_prob * advantages)) - scalar entropy_coef * mean entropy)

  let train ?(config = default) ~updates ~rngs ~policy ~critic ~distribution env =
    if updates <= 0 || config.horizon <= 0 || config.max_grad_norm <= 0.
    then invalid_arg "A2c.train: invalid configuration";
    let actor_stepper = Stepper.init (adam config.learning_rate) (Graph.mkpri policy) in
    let critic_stepper = Stepper.init (adam config.learning_rate) (Graph.mkpri critic) in
    let state = start env in
    for _ = 1 to updates do
      let rollout = collect ~horizon:config.horizon ~gamma:config.gamma ~lambda:config.lambda
          ~rng:rngs.Rng.action ~policy ~critic ~distribution env state in
      let obs, actions, _, advantages, returns = select rollout (Array.init (Array.length rollout.R.data) Fun.id) in
      let output, _ = Graph.forward policy obs in
      let loss = actor_loss ~log_prob:(distribution.Policy.log_prob output actions)
          ~advantages:(column advantages) ~entropy:(distribution.entropy output) ~entropy_coef:config.entropy_coef in
      update_clipped ~max_grad_norm:config.max_grad_norm policy actor_stepper loss;
      let predicted, _ = Graph.forward critic obs in
      update_clipped ~max_grad_norm:config.max_grad_norm critic critic_stepper
        Maths.(scalar config.value_coef * mean (sqr (predicted - returns)))
    done;
    Array.of_list (List.rev state.completed)
end
module S = Make (Owl.Neural.S.Graph)
module D = Make (Owl.Neural.D.Graph)
