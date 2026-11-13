(* RL for language models, following the progression REINFORCE -> RLOO -> PPO
   -> GRPO. A batch holds [group_size] completions for each of several
   prompts, group-major: completion [i] belongs to prompt [i / group_size].

   Policy-gradient losses are sums over token rows with one weight per row;
   the weights decide how per-token terms are aggregated into one number. *)

(* --- advantages: one number per completion, from outcome rewards --- *)

let mean xs = Array.fold_left ( +. ) 0. xs /. float_of_int (Array.length xs)

let groups ~group_size rewards =
  let n = Array.length rewards in
  if group_size <= 0 || n mod group_size <> 0 then invalid_arg "Lm_rl.groups: group size";
  Array.init (n / group_size) (fun g -> Array.sub rewards (g * group_size) group_size)

let batch_mean_baseline rewards =
  let b = mean rewards in
  Array.map (fun r -> r -. b) rewards

(* RLOO: each completion's baseline is the mean reward of the others in its group *)
let rloo ~group_size rewards =
  if group_size < 2 then invalid_arg "Lm_rl.rloo: needs at least two completions per prompt";
  let k = float_of_int (group_size - 1) in
  groups ~group_size rewards
  |> Array.map (fun g ->
         let total = Array.fold_left ( +. ) 0. g in
         Array.map (fun r -> r -. ((total -. r) /. k)) g)
  |> Array.to_list |> Array.concat

(* GRPO: standardise within the group, using the sample standard deviation *)
let grpo ?(eps = 1e-4) ~group_size rewards =
  if group_size < 2 then invalid_arg "Lm_rl.grpo: needs at least two completions per prompt";
  groups ~group_size rewards
  |> Array.map (fun g ->
         let m = mean g in
         let var = Array.fold_left (fun a r -> a +. ((r -. m) ** 2.)) 0. g /. float_of_int (group_size - 1) in
         Array.map (fun r -> (r -. m) /. (sqrt var +. eps)) g)
  |> Array.to_list |> Array.concat

(* Dr. GRPO: only subtract the group mean. Dividing by the standard deviation
   gives groups that are almost always or almost never solved larger weights. *)
let centred ~group_size rewards =
  groups ~group_size rewards
  |> Array.map (fun g ->
         let m = mean g in
         Array.map (fun r -> r -. m) g)
  |> Array.to_list |> Array.concat

let zero_groups ~group_size rewards =
  let gs = groups ~group_size rewards in
  let same = Array.fold_left (fun a g -> if Array.for_all (( = ) g.(0)) g then a + 1 else a) 0 gs in
  float_of_int same /. float_of_int (Array.length gs)

(* Fraction of ratios outside [1 - clip_low, 1 + clip_high] on the side of
   their advantage: there the clipped surrogate has no gradient. *)
let clip_fraction ~clip_low ~clip_high ~log_ratios ~advantages =
  let n = Array.length log_ratios in
  let clipped = ref 0 in
  Array.iteri
    (fun i lr ->
      let r = exp lr and a = advantages.(i) in
      if (a > 0. && r > 1. +. clip_high) || (a < 0. && r < 1. -. clip_low) then incr clipped)
    log_ratios;
  float_of_int !clipped /. float_of_int n

(* --- KL(pi || pi_ref) estimators from one token sampled from pi --- *)

(* unbiased, can be negative *)
let k1 ~log_prob ~ref_log_prob = log_prob -. ref_log_prob

(* unbiased and never negative; usually lower variance when the two are close *)
let k3 ~log_prob ~ref_log_prob =
  let d = ref_log_prob -. log_prob in
  exp d -. d -. 1.

(* how per-token terms are combined into the loss *)
type aggregation =
  | Sequence_sum (* sum over a completion's tokens, mean over completions *)
  | Sequence_mean (* mean over a completion's tokens, mean over completions *)
  | Token_mean (* mean over all tokens in the batch *)
  | Fixed_length of int (* Dr. GRPO: sum over a completion divided by a constant (the generation budget) *)

let row_weights aggregation ~lengths sequence =
  let n_seq = float_of_int (Array.length lengths) in
  let n_rows = float_of_int (Array.length sequence) in
  Array.map
    (fun s ->
      match aggregation with
      | Sequence_sum -> 1. /. n_seq
      | Sequence_mean -> 1. /. (n_seq *. float_of_int lengths.(s))
      | Token_mean -> 1. /. n_rows
      | Fixed_length l -> 1. /. (n_seq *. float_of_int l))
    sequence

module Make (Graph : Owl_neural_graph_sig.Sig) = struct
  include Lm.Make (Graph)
  open Algodiff
  module O = On_policy.Make (Graph)

  type batch =
    { problems : Lm_task.problem array (* one per completion *)
    ; samples : sample array
    ; rewards : float array (* outcome reward per completion *)
    ; lengths : int array (* tokens per completion *)
    ; contexts : int array array (* one per token row *)
    ; actions : int array
    ; sequence : int array (* completion of each row *)
    ; old_log_probs : float array (* log pi_old, recorded while sampling *)
    }

  let batch_of ~(task : Lm_task.task) problems samples =
    let contexts, actions, sequence = rows (Array.map (fun s -> s.prompt, s.tokens) samples) in
    { problems
    ; samples
    ; rewards = Array.map2 (fun p s -> task.reward p s.tokens) problems samples
    ; lengths = Array.map (fun s -> Array.length s.tokens) samples
    ; contexts
    ; actions
    ; sequence
    ; old_log_probs = Array.concat (Array.to_list (Array.map (fun s -> s.log_probs) samples))
    }

  let rollout ?(task = Lm_task.addition) ~(rngs : Rng.t) ~prompts ~group_size ~problems policy =
    let chosen = Array.init prompts (fun _ -> problems.(Random.State.int rngs.sample (Array.length problems))) in
    let problems = Array.concat (Array.to_list (Array.map (Array.make group_size) chosen)) in
    let samples =
      generate ~max_len:task.max_len ~rng:rngs.action policy (Array.map (fun (p : Lm_task.problem) -> p.prompt) problems)
    in
    batch_of ~task problems samples

  (* DAPO's dynamic sampling: sample batch after batch and keep only the groups
     whose rewards are not all equal, until [prompts] groups are kept or
     [max_rounds] batches have been sampled. Returns the kept groups, the
     first batch as sampled, and the number of completions generated. *)
  let max_rounds = 10

  let rollout_mixed ~task ~rngs ~prompts ~group_size ~problems policy =
    let kept = ref [] and n_kept = ref 0 and rounds = ref 0 and first = ref None in
    while !n_kept < prompts && !rounds < max_rounds do
      let b = rollout ~task ~rngs ~prompts ~group_size ~problems policy in
      if !first = None then first := Some b;
      incr rounds;
      Array.iteri
        (fun g rewards ->
          if !n_kept < prompts && not (Array.for_all (( = ) rewards.(0)) rewards) then (
            let sub xs = Array.sub xs (g * group_size) group_size in
            kept := (sub b.problems, sub b.samples) :: !kept;
            incr n_kept))
        (groups ~group_size b.rewards)
    done;
    let kept = List.rev !kept in
    let cat f = Array.concat (List.map f kept) in
    batch_of ~task (cat fst) (cat snd), Option.get !first, !rounds * prompts * group_size

  (* log pi(action) of every row under a network, as constants *)
  let log_probs_of net batch =
    token_log_probs (Graph.run (features batch.contexts) net) batch.actions |> unpack_arr |> values

  (* exact KL(pi || pi_ref) over the whole vocabulary at every row *)
  let exact_kl ~reference net batch =
    let x = features batch.contexts in
    let p = Distribution.Categorical.of_output (Graph.run x net)
    and q = Distribution.Categorical.of_output (Graph.run x reference) in
    Distribution.Categorical.kl p q |> unpack_arr |> values

  let per_row xs batch = Array.map (fun s -> xs.(s)) batch.sequence

  (* the mean of per-row values over each completion's tokens *)
  let sequence_means xs batch =
    let m = Array.make (Array.length batch.lengths) 0. in
    Array.iteri (fun r s -> m.(s) <- m.(s) +. (xs.(r) /. float_of_int batch.lengths.(s))) batch.sequence;
    m

  (* --- losses, all differentiable in [log_prob] --- *)

  let pg_loss ~log_prob ~advantages ~weights =
    Maths.(neg (sum' (column (Array.map2 ( *. ) weights advantages) * log_prob)))

  let clip_ratio ~clip ~clip_high ratio = Maths.(min2 (scalar (1. +. clip_high)) (max2 (scalar (1. -. clip)) ratio))

  (* PPO's clipped surrogate with per-row weights; the ratio is kept in
     [1 - clip, 1 + clip_high] (DAPO's clip-higher makes clip_high larger) *)
  let clipped_loss ~clip ~clip_high ~log_prob ~old_log_prob ~advantages ~weights =
    let ratio = Maths.(exp (log_prob - old_log_prob)) in
    let clipped = clip_ratio ~clip ~clip_high ratio in
    let a = column advantages in
    Maths.(neg (sum' (column weights * min2 (ratio * a) (clipped * a))))

  (* [n_seq; n_rows] matrix whose product with a column of per-row values
     gives the mean over each completion's tokens *)
  let sequence_mean_matrix ~lengths sequence =
    let m = A.zeros [| Array.length lengths; Array.length sequence |] in
    Array.iteri (fun r s -> A.set m [| s; r |] (A.float_to_elt (1. /. float_of_int lengths.(s)))) sequence;
    Arr m

  (* GSPO: one ratio per completion, the geometric mean of its token ratios,
     (pi_theta(y) / pi_old(y)) ^ (1 / |y|), clipped; mean over completions *)
  let sequence_clipped_loss ~clip ~clip_high ~log_prob ~old_log_prob ~advantages ~mean_matrix =
    let ratio = Maths.(exp (dot mean_matrix (log_prob - old_log_prob))) in
    let clipped = clip_ratio ~clip ~clip_high ratio in
    let a = column advantages in
    Maths.(neg (mean (min2 (ratio * a) (clipped * a))))

  (* CISPO: the clipped ratio is a constant weight on log pi, so a token whose
     ratio has left the range still has a gradient, only a capped one *)
  let cispo_loss ~clip ~clip_high ~log_prob ~old_log_prob ~advantages ~weights =
    let weight = clip_ratio ~clip ~clip_high Maths.(exp (primal' log_prob - old_log_prob)) in
    Maths.(neg (sum' (column (Array.map2 ( *. ) weights advantages) * weight * log_prob)))

  (* k3 as a loss term; its gradient pulls pi towards pi_ref *)
  let k3_loss ~log_prob ~ref_log_prob ~weights =
    let d = Maths.(ref_log_prob - log_prob) in
    Maths.(sum' (column weights * (exp d - d - scalar 1.)))

  (* the gradient of [loss] as one flat vector; the network's weights are left unchanged *)
  let flat_gradient net loss =
    let ws, gs = Graph.backward net loss in
    Graph.update net ws;
    Array.to_list gs |> List.concat_map (fun layer -> Array.to_list layer |> List.map (fun g -> values (unpack_arr g))) |> Array.concat

  type stats =
    { reward : float (* mean outcome reward of the sampled completions *)
    ; kl : float (* mean exact KL to the reference per token, 0 without one *)
    ; length : float (* mean completion length *)
    ; entropy : float (* mean exact entropy of the policy per token *)
    ; zero_groups : float (* fraction of groups whose rewards are all equal *)
    ; completions : int (* completions generated in this iteration *)
    ; clipped : float (* fraction of rows (completions for GSPO) with a clipped ratio, mean over epochs *)
    }

  (* everything except [clipped], from the batch as sampled before the update *)
  let stats ?reference ~group_size policy batch =
    let logp = Distribution.Categorical.of_output (Graph.run (features batch.contexts) policy) in
    { reward = mean batch.rewards
    ; kl = (match reference with Some r -> mean (exact_kl ~reference:r policy batch) | None -> 0.)
    ; length = mean (Array.map float_of_int batch.lengths)
    ; entropy = mean (values (unpack_arr (Distribution.Categorical.entropy logp)))
    ; zero_groups = zero_groups ~group_size batch.rewards
    ; completions = Array.length batch.samples
    ; clipped = 0.
    }

  (* log pi_theta - log pi_old of every row, as constants *)
  let log_ratios ~log_prob batch = Array.map2 ( -. ) (values (unpack_arr (primal' log_prob))) batch.old_log_probs

  let adam_clipped ~max_grad_norm ~learning_rate net =
    let stepper = Stepper.init (adam learning_rate) (Graph.mkpri net) in
    fun loss -> O.update_clipped ~max_grad_norm net stepper loss

  (* --- REINFORCE and RLOO: one gradient step per batch --- *)

  type estimator =
    | No_baseline
    | Batch_mean
    | Leave_one_out

  type pg_config =
    { prompts : int
    ; group_size : int
    ; learning_rate : float
    ; max_grad_norm : float
    ; kl_coef : float (* subtracted from the reward: beta * sum of k1 over the completion *)
    }

  let pg_default = { prompts = 64; group_size = 8; learning_rate = 1e-3; max_grad_norm = 1.; kl_coef = 0. }

  (* advantages after the optional KL penalty in the reward *)
  let pg_advantages ~config ~estimator ?reference batch =
    let rewards =
      match reference with
      | Some r when config.kl_coef > 0. ->
        let ref_lp = log_probs_of r batch in
        let penalty = Array.make (Array.length batch.rewards) 0. in
        Array.iteri
          (fun i s -> penalty.(s) <- penalty.(s) +. k1 ~log_prob:batch.old_log_probs.(i) ~ref_log_prob:ref_lp.(i))
          batch.sequence;
        Array.map2 (fun r k -> r -. (config.kl_coef *. k)) batch.rewards penalty
      | _ -> batch.rewards
    in
    match estimator with
    | No_baseline -> rewards
    | Batch_mean -> batch_mean_baseline rewards
    | Leave_one_out -> rloo ~group_size:config.group_size rewards

  let pg_gradient_loss ~config ~estimator ?reference policy batch =
    let advantages = per_row (pg_advantages ~config ~estimator ?reference batch) batch in
    let weights = row_weights Sequence_sum ~lengths:batch.lengths batch.sequence in
    let logits, _ = Graph.forward policy (features batch.contexts) in
    pg_loss ~log_prob:(token_log_probs logits batch.actions) ~advantages ~weights

  let train_pg ?(config = pg_default) ?reference ?(report = fun _ _ -> ()) ~estimator ~iterations ~rngs ~problems policy =
    if config.kl_coef > 0. && reference = None then invalid_arg "Lm_rl.train_pg: kl_coef needs a reference";
    let step = adam_clipped ~max_grad_norm:config.max_grad_norm ~learning_rate:config.learning_rate policy in
    Array.init iterations (fun i ->
        let batch = rollout ~rngs ~prompts:config.prompts ~group_size:config.group_size ~problems policy in
        let s = stats ?reference ~group_size:config.group_size policy batch in
        step (pg_gradient_loss ~config ~estimator ?reference policy batch);
        report (i + 1) s;
        s)

  (* --- GRPO: group-relative advantages, clipped ratios, KL in the loss --- *)

  (* how the policy term of the loss uses the ratio pi_theta / pi_old *)
  type objective =
    | Token_clip (* PPO's clipped surrogate on every token: GRPO, Dr. GRPO, DAPO *)
    | Sequence_clip (* GSPO: one length-normalised ratio per completion, clipped *)
    | Cispo (* CISPO: the clipped token ratio as a constant weight on log pi *)

  type grpo_config =
    { prompts : int
    ; group_size : int
    ; epochs : int (* gradient steps on each batch *)
    ; clip_low : float (* the ratio is kept in [1 - clip_low, 1 + clip_high] *)
    ; clip_high : float
    ; kl_coef : float
    ; learning_rate : float
    ; max_grad_norm : float
    ; scale_by_std : bool (* divide centred rewards by the group's standard deviation *)
    ; aggregation : aggregation
    ; objective : objective
    ; dynamic_sampling : bool (* keep only groups whose rewards differ (DAPO) *)
    ; task : Lm_task.task
    }

  let grpo_default =
    { prompts = 64; group_size = 8; epochs = 4; clip_low = 0.2; clip_high = 0.2; kl_coef = 0.1; learning_rate = 1e-3
    ; max_grad_norm = 1.; scale_by_std = true; aggregation = Sequence_mean; objective = Token_clip
    ; dynamic_sampling = false; task = Lm_task.addition }

  (* The variants, each as the switches its paper changes on a base
     configuration. None of the four papers uses a KL term: start from a base
     with kl_coef = 0 to match them. *)

  (* Dr. GRPO: no division by the standard deviation or by the completion's length *)
  let dr_grpo c = { c with scale_by_std = false; aggregation = Fixed_length c.task.max_len }

  (* DAPO: clip-higher, a token-level mean, dynamic sampling (its overlong
     reward shaping is not implemented) *)
  let dapo c = { c with clip_low = 0.2; clip_high = 0.28; aggregation = Token_mean; dynamic_sampling = true }

  (* GSPO, with the paper's clipping range for sequence ratios *)
  let gspo c = { c with objective = Sequence_clip; clip_low = 3e-4; clip_high = 4e-4 }

  (* CISPO: no lower bound on the weight, a token-level mean; the paper tunes
     clip_high without giving a value, so the base's is kept *)
  let cispo c = { c with objective = Cispo; clip_low = 1.; aggregation = Token_mean }

  (* Three policies: pi_old generated the batch (its log-probs are stored),
     pi_theta is being optimised, pi_ref is the frozen starting point. *)
  let train_grpo ?(config = grpo_default) ?(report = fun _ _ -> ()) ~reference ~iterations ~rngs ~problems policy =
    let step = adam_clipped ~max_grad_norm:config.max_grad_norm ~learning_rate:config.learning_rate policy in
    let { task; prompts; group_size; clip_low = clip; clip_high; _ } = config in
    Array.init iterations (fun i ->
        let batch, s =
          if config.dynamic_sampling then (
            let batch, first, completions = rollout_mixed ~task ~rngs ~prompts ~group_size ~problems policy in
            batch, { (stats ~reference ~group_size policy first) with completions })
          else (
            let batch = rollout ~task ~rngs ~prompts ~group_size ~problems policy in
            batch, stats ~reference ~group_size policy batch)
        in
        (* dynamic sampling may keep no group at all *)
        if Array.length batch.actions = 0 then (report (i + 1) s; s)
        else (
          let advantages = if config.scale_by_std then grpo ~group_size batch.rewards else centred ~group_size batch.rewards in
          let row_advantages = per_row advantages batch in
          let weights = row_weights config.aggregation ~lengths:batch.lengths batch.sequence in
          let old_log_prob = column batch.old_log_probs in
          let ref_log_prob = lazy (column (log_probs_of reference batch)) in
          let mean_matrix = lazy (sequence_mean_matrix ~lengths:batch.lengths batch.sequence) in
          let x = features batch.contexts in
          let clipped = ref 0. in
          for _ = 1 to config.epochs do
            let logits, _ = Graph.forward policy x in
            let log_prob = token_log_probs logits batch.actions in
            let log_ratios = log_ratios ~log_prob batch in
            let policy_loss, fraction =
              match config.objective with
              | Token_clip ->
                ( clipped_loss ~clip ~clip_high ~log_prob ~old_log_prob ~advantages:row_advantages ~weights
                , clip_fraction ~clip_low:clip ~clip_high ~log_ratios ~advantages:row_advantages )
              | Cispo ->
                ( cispo_loss ~clip ~clip_high ~log_prob ~old_log_prob ~advantages:row_advantages ~weights
                , clip_fraction ~clip_low:clip ~clip_high ~log_ratios ~advantages:row_advantages )
              | Sequence_clip ->
                ( sequence_clipped_loss ~clip ~clip_high ~log_prob ~old_log_prob ~advantages ~mean_matrix:(Lazy.force mean_matrix)
                , clip_fraction ~clip_low:clip ~clip_high ~log_ratios:(sequence_means log_ratios batch) ~advantages )
            in
            clipped := !clipped +. (fraction /. float_of_int config.epochs);
            let loss =
              if config.kl_coef > 0. then
                Maths.(policy_loss + (scalar config.kl_coef * k3_loss ~log_prob ~ref_log_prob:(Lazy.force ref_log_prob) ~weights))
              else policy_loss
            in
            step loss
          done;
          let s = { s with clipped = !clipped } in
          report (i + 1) s;
          s))

  (* --- PPO at the token level: a critic, per-token KL penalty, GAE --- *)

  type ppo_config =
    { prompts : int
    ; group_size : int (* completions per prompt, only to match GRPO's sampling budget *)
    ; epochs : int
    ; clip : float
    ; kl_coef : float (* per-token penalty -beta * k1 in the reward *)
    ; gamma : float
    ; lambda : float
    ; learning_rate : float
    ; max_grad_norm : float
    }

  let ppo_default =
    { prompts = 64; group_size = 8; epochs = 4; clip = 0.2; kl_coef = 0.1; gamma = 1.; lambda = 0.95
    ; learning_rate = 1e-3; max_grad_norm = 1. }

  (* Token rewards: -beta * k1 at every token, plus the outcome reward at the
     last one. Generation ends after the last token either way (eos or the
     length limit), so every completion ends in a terminal state. *)
  let token_advantages ~config ~ref_log_probs ~values batch =
    let n = Array.length batch.actions in
    let last i = i = n - 1 || batch.sequence.(i + 1) <> batch.sequence.(i) in
    let rewards =
      Array.init n (fun i ->
          (-.config.kl_coef *. k1 ~log_prob:batch.old_log_probs.(i) ~ref_log_prob:ref_log_probs.(i))
          +. if last i then batch.rewards.(batch.sequence.(i)) else 0.)
    in
    let terminated = Array.init n last in
    let next_values = Array.init n (fun i -> if last i then 0. else values.(i + 1)) in
    Rollout.gae ~n_envs:1 ~gamma:config.gamma ~lambda:config.lambda ~rewards ~values ~next_values ~terminated
      ~truncated:(Array.make n false)

  let train_ppo ?(config = ppo_default) ?(report = fun _ _ -> ()) ~reference ~critic ~iterations ~rngs ~problems policy =
    let actor_step = adam_clipped ~max_grad_norm:config.max_grad_norm ~learning_rate:config.learning_rate policy in
    let critic_step = adam_clipped ~max_grad_norm:config.max_grad_norm ~learning_rate:config.learning_rate critic in
    Array.init iterations (fun i ->
        let batch = rollout ~rngs ~prompts:config.prompts ~group_size:config.group_size ~problems policy in
        let s = stats ~reference ~group_size:config.group_size policy batch in
        let x = features batch.contexts in
        let values = Graph.run x critic |> unpack_arr |> values in
        let advantages, returns = token_advantages ~config ~ref_log_probs:(log_probs_of reference batch) ~values batch in
        let advantages = O.normalise advantages in
        let weights = row_weights Token_mean ~lengths:batch.lengths batch.sequence in
        let old_log_prob = column batch.old_log_probs in
        let clipped = ref 0. in
        for _ = 1 to config.epochs do
          let logits, _ = Graph.forward policy x in
          let log_prob = token_log_probs logits batch.actions in
          clipped :=
            !clipped
            +. clip_fraction ~clip_low:config.clip ~clip_high:config.clip ~log_ratios:(log_ratios ~log_prob batch) ~advantages
               /. float_of_int config.epochs;
          actor_step (clipped_loss ~clip:config.clip ~clip_high:config.clip ~log_prob ~old_log_prob ~advantages ~weights);
          let v, _ = Graph.forward critic x in
          critic_step Maths.(mean (sqr (v - column returns)))
        done;
        let s = { s with clipped = !clipped } in
        report (i + 1) s;
        s)
end

module S = Make (Owl.Neural.S.Graph)
module D = Make (Owl.Neural.D.Graph)
