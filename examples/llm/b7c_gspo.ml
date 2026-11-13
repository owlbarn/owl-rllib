(* B7c. GSPO: one importance ratio per completion.

   GRPO weighs every token by its own ratio pi_theta / pi_old, an importance
   weight estimated from a single sample. GSPO uses one ratio per completion,
   the geometric mean of its token ratios, (pi_theta(y) / pi_old(y))^(1/|y|),
   and clips that. Being a mean, it is much closer to 1, and the paper clips
   it to [1 - 3e-4, 1 + 4e-4] instead of [1 - 0.2, 1 + 0.2].

   1. After the 4 updates on one batch: how far token and sequence ratios
      have moved, on 3-4 token completions and on the longer ones with filler.
   2. Training with the paper's range and with 0.2.

   OPENBLAS_NUM_THREADS=1 dune exec examples/llm/b7c_gspo.exe  (about 3 minutes) *)
open Owl_rllib
open Owl_rllib_llm
open Common

let std xs =
  let m = Lm_rl.mean xs in
  sqrt (Lm_rl.mean (Array.map (fun x -> (x -. m) ** 2.) xs))

let fraction p xs = float_of_int (Array.fold_left (fun a x -> if p x then a + 1 else a) 0 xs) /. float_of_int (Array.length xs)

let () =
  let iterations = ref 250 in
  Arg.parse [ "--iterations", Arg.Set_int iterations, "Training iterations" ] (fun _ -> ()) "B7c";

  (* 1. Ratios after one batch of GRPO updates; the same rng seed makes
     train_grpo sample the same batch *)
  print_endline "log pi_theta / pi_old after 4 GRPO updates on one batch of 64 x 8:";
  Printf.printf "  %-22s %11s %13s %18s %18s\n" "task" "mean length" "std: token" "sequence" "|seq| > 4e-4";
  List.iter
    (fun (name, task) ->
      let start = fine_tuned ~task () in
      let config = { grpo_base with task } in
      let batch = R.rollout ~task ~rngs:(Rng.create 3) ~prompts:config.prompts ~group_size:config.group_size ~problems:train start in
      let policy = R.Graph.copy start in
      ignore (R.train_grpo ~config ~reference:start ~iterations:1 ~rngs:(Rng.create 3) ~problems:train policy);
      let token = Array.map2 ( -. ) (R.log_probs_of policy batch) batch.old_log_probs in
      let sequence = R.sequence_means token batch in
      Printf.printf "  %-22s %11.2f %13.4f %18.4f %18.2f\n%!" name (Lm_rl.mean (Array.map float_of_int batch.lengths))
        (std token) (std sequence) (fraction (fun x -> Float.abs x > 4e-4) sequence))
    [ "addition", Lm_task.addition; "addition with filler", Lm_task.addition_with_filler ];
  print_endline "Sequence ratios spread about half as much as token ratios. The paper's range was chosen for its";
  print_endline "own models and learning rate; with this model's updates it leaves out almost every completion.\n";

  (* 2. Training *)
  let start = fine_tuned () in
  List.iter
    (fun (label, config) -> ignore (train_variant ~iterations:!iterations label config start))
    [ "GRPO (no KL)", grpo_base
    ; "GSPO, range [1 - 3e-4, 1 + 4e-4]", R.gspo grpo_base
    ; "GSPO, range [1 - 0.2, 1 + 0.2]", { (R.gspo grpo_base) with clip_low = 0.2; clip_high = 0.2 } ]
