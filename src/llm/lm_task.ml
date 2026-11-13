(* A verifiable toy task for language-model RL: two-digit addition.

   The prompt "27+48=" asks for the completion "75" followed by an
   end-of-sequence token. Answers have two or three digits, so completions
   have different lengths. *)

let plus = 10
let equals = 11
let eos = 12
let pad = 13
let vocab = 14

type problem = { a : int; b : int; prompt : int array; answer : int array (* digits, then eos *) }

let digits n = String.to_seq (string_of_int n) |> Seq.map (fun c -> Char.code c - 48) |> Array.of_seq

let problem a b =
  { a; b
  ; prompt = Array.concat [ digits a; [| plus |]; digits b; [| equals |] ]
  ; answer = Array.append (digits (a + b)) [| eos |] }

(* every pair of operands in [10, 99], shuffled with [seed] and split in two *)
let split ?(test_fraction = 0.2) ~seed () =
  let all = Array.init (90 * 90) (fun i -> problem (10 + (i / 90)) (10 + (i mod 90))) in
  let order = Rollout.permutation ~rng:(Random.State.make [| seed |]) (Array.length all) in
  let n_test = int_of_float (test_fraction *. float_of_int (Array.length all)) in
  let shuffled = Array.map (Array.get all) order in
  Array.sub shuffled n_test (Array.length all - n_test), Array.sub shuffled 0 n_test

(* longest completion the verifier can accept *)
let max_answer = 4

(* 1 for exactly the right digits followed by eos, 0 otherwise *)
let outcome_reward p completion = if completion = p.answer then 1. else 0.

(* one reward per generated token: 1 / |answer| for each token in the right
   place, so a perfect completion still earns 1 in total *)
let process_rewards p completion =
  let n = Array.length p.answer in
  Array.mapi (fun i t -> if i < n && p.answer.(i) = t then 1. /. float_of_int n else 0.) completion

(* What generation and fine-tuning need to know about a task: the length
   limit, the verifier, and the completion SFT imitates. *)
type task =
  { max_len : int
  ; reward : problem -> int array -> float
  ; target : Random.State.t -> problem -> int array
  }

let addition = { max_len = max_answer; reward = outcome_reward; target = (fun _ p -> p.answer) }

(* For length-bias experiments (examples/llm/b7a_dr_grpo.ml). Up to [max_filler] filler tokens (pad) may follow
   the digits before eos and the verifier ignores them, so completion length
   varies without affecting the reward: any drift in length comes from the
   training objective. SFT imitates a uniformly random amount of filler.
   Contexts then grow beyond Lm's window and lose the start of the prompt,
   but only after the digits have been written. *)
let max_filler = 4

let without_filler completion = Array.of_list (List.filter (( <> ) pad) (Array.to_list completion))

let addition_with_filler =
  { max_len = max_answer + max_filler
  ; reward = (fun p completion -> outcome_reward p (without_filler completion))
  ; target =
      (fun rng p ->
        let digits = Array.sub p.answer 0 (Array.length p.answer - 1) in
        Array.concat [ digits; Array.make (Random.State.int rng (max_filler + 1)) pad; [| eos |] ])
  }

let to_string tokens =
  Array.map
    (fun t -> if t < 10 then string_of_int t else if t = plus then "+" else if t = equals then "=" else if t = eos then "$" else "_")
    tokens
  |> Array.to_list |> String.concat ""
