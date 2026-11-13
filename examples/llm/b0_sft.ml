(* B0. Supervised fine-tuning: the starting point for RL.

   A toy language model learns two-digit addition from correct examples
   (teacher forcing, cross-entropy on every answer token). We stop early on
   purpose, so that the model is right only part of the time and RL has
   something to improve.

   OPENBLAS_NUM_THREADS=1 dune exec examples/llm/b0_sft.exe *)
open Owl_rllib_llm
open Common

let () =
  let steps = ref 2000 in
  Arg.parse [ "--steps", Arg.Set_int steps, "Fine-tuning steps to show" ] (fun _ -> ()) "B0";
  Printf.printf "%d training problems, %d test problems, vocabulary of %d tokens, window of %d tokens\n\n"
    (Array.length train) (Array.length test) Lm_task.vocab R.window;
  let net = R.create ~seed:1 () in
  print_eval "before fine-tuning" net;
  R.sft ~steps:!steps ~rng:(Random.State.make [| 1 |]) net train ~report:(fun step loss ->
      if step mod 250 = 0 then (
        Printf.printf "step %4d  loss %.3f  " step loss;
        print_eval (if step = sft_steps then "<- the B chapters start here" else "") net));
  print_endline "\nSampled completions from the model after the last step:";
  show_completions ~rng:(Random.State.make [| 2 |]) net;
  print_endline "\nAnd from the starting point of the B chapters:";
  show_completions ~rng:(Random.State.make [| 2 |]) (fine_tuned ())
