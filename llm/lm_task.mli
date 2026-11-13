(** A verifiable toy task for language-model RL: two-digit addition.

    The prompt ["27+48="] asks for the completion ["75"] followed by an end-of-sequence token.
    Answers have two or three digits, so completions have different lengths. Tokens: the digits 0-9,
    then [plus], [equals], [eos] and [pad]. *)

val plus : int
val equals : int
val eos : int
val pad : int
val vocab : int

type problem = {
  a : int;
  b : int;
  prompt : int array;
  answer : int array;  (** digits, then [eos] *)
}

val problem : int -> int -> problem

val split : ?test_fraction:float -> seed:int -> unit -> problem array * problem array
(** All pairs of operands in [[10, 99]], shuffled with [seed] and split into training and test sets
    ([test_fraction] defaults to 0.2). *)

val max_answer : int
(** The longest completion the verifier can accept. *)

val outcome_reward : problem -> int array -> float
(** 1 for exactly the right digits followed by [eos], 0 otherwise. *)

val process_rewards : problem -> int array -> float array
(** One reward per generated token: [1 / |answer|] for each token in the right place, so a perfect
    completion still earns 1 in total. *)

type task = {
  max_len : int;  (** generation stops after this many tokens *)
  reward : problem -> int array -> float;  (** the verifier *)
  target : Random.State.t -> problem -> int array;  (** the completion SFT imitates *)
}
(** What generation and fine-tuning need to know about a task. *)

val addition : task
val max_filler : int
val without_filler : int array -> int array

val addition_with_filler : task
(** Up to [max_filler] filler tokens ([pad]) may follow the digits before [eos]; the verifier
    ignores them, so a completion's length does not affect its reward and any drift in length comes
    from the training objective. SFT imitates a uniformly random amount of filler. *)

val to_string : int array -> string
(** Tokens as text, with [$] for [eos] and [_] for [pad]. *)
