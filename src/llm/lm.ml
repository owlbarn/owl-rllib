(* A toy autoregressive language model: an MLP that reads the last [window]
   tokens, one-hot encoded, and outputs logits for the next token. One-hot
   input followed by a linear layer is a learned token embedding, one per
   position. Every generated position is one row of a batch. *)
module Make (Graph : Owl_neural_graph_sig.Sig) = struct
  include Neural.Make (Graph)
  open Algodiff

  let vocab = Lm_task.vocab

  (* longest prompt "99+99=" plus every answer token except the last one *)
  let window = 6 + Lm_task.max_answer - 1
  let input_size = window * vocab

  (* left-padded one-hot window, row-major [position; token] *)
  let encode context =
    let x = Array.make input_size 0. in
    let len = Array.length context in
    for k = 0 to window - 1 do
      let j = len - window + k in
      let t = if j < 0 then Lm_task.pad else context.(j) in
      x.((k * vocab) + t) <- 1.
    done;
    x

  let features contexts =
    let flat = Array.concat (Array.to_list (Array.map encode contexts)) |> Array.map A.float_to_elt in
    Arr (A.of_array flat [| Array.length contexts; input_size |])

  let create ?(hidden = [ 128; 128 ]) ?(outputs = vocab) ~seed () =
    mlp ~seed ~shape:[| input_size |] ~hidden ~outputs

  (* log pi(action | context) for every row, differentiable through [logits] *)
  let token_log_probs logits actions =
    Distribution.Categorical.log_prob (Distribution.Categorical.of_output logits) (one_hot vocab actions)

  type sample =
    { prompt : int array
    ; tokens : int array (* generated tokens, ending with eos unless cut off *)
    ; log_probs : float array (* log pi(token) under the generating network *)
    }

  (* Generate up to [max_len] tokens for every prompt, one batched forward pass
     per position. Greedy decoding takes the most likely token. *)
  let generate ?(greedy = false) ?(max_len = Lm_task.max_answer) ~rng net prompts =
    let n = Array.length prompts in
    let tokens = Array.make n [] and logps = Array.make n [] in
    let finished = Array.make n false in
    for _ = 1 to max_len do
      let active = List.filter (fun i -> not finished.(i)) (List.init n Fun.id) |> Array.of_list in
      if Array.length active > 0
      then (
        let contexts = Array.map (fun i -> Array.append prompts.(i) (Array.of_list (List.rev tokens.(i)))) active in
        let logp = Graph.run (features contexts) net |> Distribution.Categorical.of_output |> unpack_arr in
        Array.iteri
          (fun row i ->
            let lp = Array.init vocab (fun v -> A.get logp [| row; v |] |> A.elt_to_float) in
            let t = if greedy then Dp.argmax lp else Mdp.sample_from rng (List.init vocab (fun v -> v, exp lp.(v))) in
            tokens.(i) <- t :: tokens.(i);
            logps.(i) <- lp.(t) :: logps.(i);
            if t = Lm_task.eos then finished.(i) <- true)
          active)
    done;
    Array.init n (fun i ->
        { prompt = prompts.(i); tokens = Array.of_list (List.rev tokens.(i)); log_probs = Array.of_list (List.rev logps.(i)) })

  (* one row per generated token: the context before it and the token itself *)
  let rows (completions : (int array * int array) array) =
    let contexts = ref [] and actions = ref [] and sequence = ref [] in
    Array.iteri
      (fun s (prompt, tokens) ->
        Array.iteri
          (fun t token ->
            contexts := Array.append prompt (Array.sub tokens 0 t) :: !contexts;
            actions := token :: !actions;
            sequence := s :: !sequence)
          tokens)
      completions;
    let rev l = Array.of_list (List.rev l) in
    rev !contexts, rev !actions, rev !sequence

  (* fraction of problems answered exactly right, with greedy decoding or with
     one sample per problem (the quantity RL maximises); the sampling stream is
     fixed so that evaluations are repeatable *)
  let accuracy ?(sample = false) ?(task = Lm_task.addition) net (problems : Lm_task.problem array) =
    let samples =
      generate ~greedy:(not sample) ~max_len:task.max_len ~rng:(Random.State.make [| 0 |]) net
        (Array.map (fun (p : Lm_task.problem) -> p.prompt) problems)
    in
    let correct = Array.map2 (fun p s -> task.reward p s.tokens) problems samples in
    Array.fold_left ( +. ) 0. correct /. float_of_int (Array.length problems)

  (* Supervised fine-tuning on the correct answers (teacher forcing): minimise
     the cross-entropy of every answer token given the tokens before it. *)
  let sft ?(batch_size = 128) ?(learning_rate = 1e-3) ?(report = fun _ _ -> ()) ?(task = Lm_task.addition) ~steps ~rng
      net (problems : Lm_task.problem array)
    =
    let stepper = Stepper.init (adam learning_rate) (Graph.mkpri net) in
    for step = 1 to steps do
      let batch = Array.init batch_size (fun _ -> problems.(Random.State.int rng (Array.length problems))) in
      let contexts, actions, _ = rows (Array.map (fun (p : Lm_task.problem) -> p.prompt, task.target rng p) batch) in
      let logits, _ = Graph.forward net (features contexts) in
      let loss = Maths.(neg (mean (token_log_probs logits actions))) in
      let value = unpack_flt loss in
      update net stepper loss;
      report step value
    done
end

module S = Make (Owl.Neural.S.Graph)
module D = Make (Owl.Neural.D.Graph)
