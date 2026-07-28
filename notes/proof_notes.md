# Proof notes: Decision-Relevant Memory Selection

This document records the theorem stack used by the paper.  It is written more
explicitly than the main text so that every dependence and proof obligation is
visible.

## 1. Setting

Consider a finite-horizon controlled process with decision stages
\(h\in[H]\).  The *full observable context* at stage \(h\), denoted
\(X_h\in\mathcal X_h\), is the complete action--observation history available
to the learner.  Treating the complete history as the state makes the process a
finite-horizon MDP, with reward \(r_h(x,a)\in[0,1]\) and transition kernel
\(P_h(\cdot\mid x,a)\).  No candidate truncated history is assumed Markov.

For a candidate memory length \(m\), let \(\phi_{m,h}:\mathcal X_h\to
\mathcal Z_{m,h}\) retain the most recent \(m\) observations and the
intervening actions (or the entire history when \(h<m\)).  The corresponding
policy class is

\[
\Pi_m=\left\{\pi:\pi_h(\cdot\mid x)=\pi_h(\cdot\mid
\phi_{m,h}(x))\right\}.
\]

The classes are nested: \(\Pi_1\subseteq\cdots\subseteq\Pi_M\).
Let \(Q_h^*,V_h^*\) be the optimal full-context action-value and value
functions and define the nonnegative optimal advantage loss

\[
A_h^*(x,a)=V_h^*(x)-Q_h^*(x,a).
\]

For a suffix \(z\), write
\(C_{m,h}(z)=\{x:\phi_{m,h}(x)=z\}\).

## 2. Population decision ambiguity

### Definition 1 (cell and stage ambiguity)

For every nonempty suffix cell,

\[
\eta_{m,h}(z)
=\min_{p\in\Delta(\mathcal A_h)}
\max_{x\in C_{m,h}(z)}
\langle p,A_h^*(x,\cdot)\rangle .
\]

Define

\[
\eta_{m,h}=\max_{z\in\mathcal Z_{m,h}}\eta_{m,h}(z),
\qquad
G_m=\sum_{h=1}^H\eta_{m,h}.
\]

The minimizer \(p_{m,h,z}\) is a common action distribution used at every full
context in the cell.  It is obtained by a linear program with \(|\mathcal A_h|
+1\) variables and \(|C_{m,h}(z)|\) inequality constraints.

### Lemma 1 (monotonicity)

For every \(h\), \(\eta_{m+1,h}\leq\eta_{m,h}\), and hence
\(G_{m+1}\leq G_m\).

**Proof.** Every \((m+1)\)-suffix cell is contained in an \(m\)-suffix cell.
Fix a refined cell \(C'\subseteq C\), and let \(p_C\) minimize the objective
on \(C\).  Then

\[
\eta(C')\leq \max_{x\in C'}\langle p_C,A_h^*(x,\cdot)\rangle
\leq \max_{x\in C}\langle p_C,A_h^*(x,\cdot)\rangle=\eta(C).
\]

Taking maxima over refined cells and then summing over stages proves the
claim. \(\square\)

### Theorem 1 (exact decision sufficiency)

The following are equivalent for a fixed \(m\):

1. \(G_m=0\).
2. Every suffix cell has a common optimal action distribution:
   there is \(p_{h,z}\) such that
   \(\langle p_{h,z},A_h^*(x,\cdot)\rangle=0\) for all
   \(x\in C_{m,h}(z)\).
3. There exists \(\pi\in\Pi_m\) satisfying
   \(V_h^\pi(x)=V_h^*(x)\) for every stage and full context.

For deterministic policies, condition 2 reduces to

\[
\bigcap_{x\in C_{m,h}(z)}\arg\max_a Q_h^*(x,a)\neq\varnothing
\quad\text{for every }h,z.
\]

**Proof.** Since all optimal advantage losses are nonnegative,
\(\eta_{m,h}(z)=0\) if and only if its minimizing distribution has zero
expected loss at every context in the cell.  This proves 1 \(\Leftrightarrow\)
2.  Under 2, define a suffix policy using these distributions.  Backward
induction gives

\[
V_h^\pi(x)
=\mathbb E_{a\sim p_{h,z}}[r_h(x,a)+P_hV_{h+1}^\pi(x,a)]
=\mathbb E_{a\sim p_{h,z}}Q_h^*(x,a)=V_h^*(x),
\]

where the second equality uses the induction hypothesis.  Thus 2 implies 3.
Conversely, if \(\pi\in\Pi_m\) is uniformly optimal, then the same backward
induction implies
\(V_h^*(x)=\mathbb E_{a\sim\pi_h(z)}Q_h^*(x,a)\), so its expected optimal
advantage loss is zero in every cell. \(\square\)

Define the best uniform value loss attainable by the class

\[
\mathcal E_m
=\inf_{\pi\in\Pi_m}\max_{h\in[H],x\in\mathcal X_h}
\{V_h^*(x)-V_h^\pi(x)\}.
\]

### Theorem 2 (value-loss sandwich)

\[
\boxed{
\max_{h\in[H]}\eta_{m,h}
\;\leq\;
\mathcal E_m
\;\leq\;
G_m.
}
\]

Moreover, the cellwise minimax policy \(\pi^m\) obeys the stagewise bound

\[
V_h^*(x)-V_h^{\pi^m}(x)
\leq \sum_{t=h}^H\eta_{m,t}
\quad\text{for all }h,x.
\]

**Upper-bound proof.** For \(z=\phi_{m,h}(x)\), decompose the loss as

\[
\begin{aligned}
D_h(x)
&=V_h^*(x)-V_h^{\pi^m}(x)\\
&=\mathbb E_{a\sim p_{m,h,z}}A_h^*(x,a)
 +\mathbb E_{a\sim p_{m,h,z},X'\sim P_h(\cdot\mid x,a)}D_{h+1}(X').
\end{aligned}
\]

The first term is at most \(\eta_{m,h}\), and the second is at most
\(\|D_{h+1}\|_\infty\).  Backward induction proves the stagewise inequality
and therefore \(\mathcal E_m\leq G_m\).

**Lower-bound proof.** For any \(\pi\in\Pi_m\),

\[
V_h^\pi(x)
=\mathbb E_{a\sim\pi_h(z)}[r_h(x,a)+P_hV_{h+1}^\pi(x,a)]
\leq \mathbb E_{a\sim\pi_h(z)}Q_h^*(x,a),
\]

because \(V_{h+1}^\pi\leq V_{h+1}^*\).  Hence

\[
V_h^*(x)-V_h^\pi(x)
\geq \langle\pi_h(z),A_h^*(x,\cdot)\rangle.
\]

Maximizing within each suffix cell and minimizing over the shared action
distribution yields at least \(\eta_{m,h}\).  Taking the maximum over stages
and then the infimum over policies proves the lower bound. \(\square\)

## 3. Observation prediction versus decision sufficiency

Fix a behavior policy \(\mu\) and define the oracle next-observation residual

\[
R_m^{\mathrm{obs}}
=
\sum_{h=1}^H H_\mu(O_{h+1}\mid\phi_{m,h}(X_h),A_h)
-
\sum_{h=1}^H H_\mu(O_{h+1}\mid X_h,A_h).
\]

This is deliberately an **observation-prediction diagnostic**, matching the
kind of score used by recent prediction-based Markov-violation methods.  It is
not the full controlled Markov condition, which also includes rewards and the
next decision state.

### Proposition 1 (decision-only memory for observation prediction)

For every \(H\geq2\) and gap \(\gamma>0\), there is a process in which a binary
cue appears only at stage one, all subsequent observations are constant, and
the final rewarding action equals the cue.  Then
\(R_m^{\mathrm{obs}}=0\) for every \(m\), while

\[
G_m=\gamma/2\quad (m<H),\qquad G_H=0
\]

under a balanced cue.  Thus a next-observation diagnostic can miss memory that
is essential for action choice.

### Proposition 2 (prediction-only memory)

For every \(H\geq2\), there is a process in which the stage-one cue reappears in
the terminal observation but action zero is optimal after both cues.  Then
\(G_m=0\) for every \(m\), while \(R_m^{\mathrm{obs}}=1\) bit for
\(m<H\) under a balanced cue and zero at \(m=H\).  Observation-predictive and
decision-relevant orders can therefore differ in either direction by an
arbitrarily long horizon.

### Proposition 3 (full controlled predictive sufficiency is stronger)

Suppose that for every stage, suffix cell, and action, the conditional joint
law of

\[
(R_h,\phi_{m,h+1}(X_{h+1}))
\]

is identical for all full histories in the cell.  Then \(G_m=0\).

**Proof.** At the terminal stage, equal reward laws imply that
\(Q_H^*(x,a)\) is constant within each suffix cell for every action.  Suppose
the claim holds at stage \(h+1\).  The joint-law condition and the induction
hypothesis imply that

\[
Q_h^*(x,a)
=\mathbb E[R_h+V_{h+1}^*(X_{h+1})\mid x,a]
\]

is constant within each current suffix cell for each action.  Hence every cell
has a common maximizing action and zero ambiguity. \(\square\)

The two-sided separation therefore concerns observation-only predictive
proxies, not an exact reward--transition Markov representation.  Exact
controlled predictive order upper-bounds decision-relevant order and can be
strictly larger.

Both toy constructions are encoded in `src/drms/toy_envs.py`.

## 4. High-confidence offline certificate

Assume simultaneous intervals

\[
L_h(x,a)\leq Q_h^*(x,a)\leq U_h(x,a)
\quad\text{for all }h,x,a
\]

hold on an event \(\mathcal E\).  Define the pairwise upper optimal-advantage
cost

\[
\overline A_h(x,a)
=\max\left\{0,\max_{b\neq a}[U_h(x,b)-L_h(x,a)]\right\}.
\]

When only one action is available, set \(\overline A_h(x,a)=0\).  Excluding the
self-comparison is important: the trivial comparison of action \(a\) with
itself is exactly zero, not \(U_h(x,a)-L_h(x,a)\).

On \(\mathcal E\),

\[
A_h^*(x,a)
=\max\left\{0,\max_{b\neq a}[Q_h^*(x,b)-Q_h^*(x,a)]\right\}
\leq \overline A_h(x,a).
\]

Replace \(A_h^*\) by \(\overline A_h\) in Definition 1 to obtain robust cell,
stage, and cumulative ambiguities \(\overline\eta_{m,h}(z)\),
\(\overline\eta_{m,h}\), and
\(\overline G_m=\sum_h\overline\eta_{m,h}\).  Let
\(\overline\pi^m\) be the corresponding cellwise minimax policy.

### Theorem 3 (simultaneous value certificate)

On \(\mathcal E\), simultaneously for every candidate memory \(m\),

\[
V_h^*(x)-V_h^{\overline\pi^m}(x)
\leq\sum_{t=h}^H\overline\eta_{m,t}
\quad\text{for every }h,x.
\]

Consequently, if DRMS selects

\[
\widehat m_\epsilon
=\min\{m:\overline G_m\leq\epsilon\},
\]

then

\[
\max_x[V_1^*(x)-V_1^{\overline\pi^{\widehat m_\epsilon}}(x)]
\leq\epsilon.
\]

If the set is empty, returning the longest memory with an explicit
`certified=False` flag is a valid abstention rather than a false claim.

**Proof.** Repeat the recursion in Theorem 2 and replace the first term by its
upper bound:

\[
\mathbb E_{a\sim\overline\pi_h^m(z)}A_h^*(x,a)
\leq
\mathbb E_{a\sim\overline\pi_h^m(z)}\overline A_h(x,a)
\leq\overline\eta_{m,h}.
\]

Backward induction proves the result.  Because the interval event is
simultaneous, selecting \(m\) using the same intervals does not require a
separate post-selection correction. \(\square\)

### Proposition 3 (certificate tightness under uniform interval error)

Suppose on \(\mathcal E\)

\[
Q_h^*(x,a)-w\leq L_h(x,a)\leq Q_h^*(x,a)
\leq U_h(x,a)\leq Q_h^*(x,a)+w
\]

for every \(h,x,a\).  Then

\[
A_h^*(x,a)\leq\overline A_h(x,a)\leq A_h^*(x,a)+2w,
\]

and therefore

\[
\eta_{m,h}\leq\overline\eta_{m,h}\leq\eta_{m,h}+2w,
\qquad
G_m\leq\overline G_m\leq G_m+2Hw.
\]

The upper bound follows by evaluating the robust LP at a minimizer of the exact
LP.

### Corollary 1 (exact memory recovery)

If

\[
m_{\rm dec}=\min\{m:G_m=0\}>1,
\qquad
\Gamma=\min_{m<m_{\rm dec}}G_m,
\]

and \(2Hw<\Gamma\), DRMS using any threshold
\(\epsilon\in[2Hw,\Gamma)\) satisfies \(\widehat m_\epsilon=m_{\rm dec}\) on
\(\mathcal E\).

## 5. A transparent tabular confidence-set instantiation

Let \(N_h(x,a)\) be the number of visits in \(n\) independent logged episodes,
and let \(M=\sum_h|\mathcal X_h||\mathcal A_h|\).  The implementation uses

\[
\beta^r_h(x,a)
=\sqrt{\frac{\log(4M/\delta)}{2\max\{1,N_h(x,a)\}}}
\]

(with radius one for an unobserved pair), and the Weissman-style transition
radius

\[
\beta^P_h(x,a)
=\min\left\{2,
\sqrt{\frac{2\{ |\mathcal X_{h+1}|\log2+\log(2M/\delta)\}}
{\max\{1,N_h(x,a)\}}}
\right\}.
\]

A union bound gives, with probability at least \(1-\delta\), simultaneous
reward and transition concentration:

\[
|\widehat r_h-r_h|\leq\beta^r_h,
\qquad
\|\widehat P_h-P_h\|_1\leq\beta^P_h.
\]

Set \(\underline V_{H+1}=\overline V_{H+1}=0\), and recurse backward:

\[
\begin{aligned}
\underline Q_h
&=\operatorname{clip}_{[0,H-h+1]}\left(
\widehat r_h-\beta_h^r
+\widehat P_h\underline V_{h+1}
-\tfrac12\beta_h^P\operatorname{span}(\underline V_{h+1})
\right),\\
\overline Q_h
&=\operatorname{clip}_{[0,H-h+1]}\left(
\widehat r_h+\beta_h^r
+\widehat P_h\overline V_{h+1}
+\tfrac12\beta_h^P\operatorname{span}(\overline V_{h+1})
\right),\\
\underline V_h&=\max_a\underline Q_h(\cdot,a),
\qquad
\overline V_h=\max_a\overline Q_h(\cdot,a).
\end{aligned}
\]

### Proposition 4 (validity of tabular intervals)

On the simultaneous model-concentration event,

\[
\underline Q_h(x,a)\leq Q_h^*(x,a)\leq\overline Q_h(x,a)
\quad\text{for every }h,x,a.
\]

**Proof.** Use backward induction and the variation inequality

\[
|(P-\widehat P)^\top f|
\leq\tfrac12\|P-\widehat P\|_1\operatorname{span}(f).
\]

For the lower bound, \(V_{h+1}^*\geq\underline V_{h+1}\), so

\[
Q_h^*=r_h+P_hV_{h+1}^*
\geq (\widehat r_h-\beta_h^r)
+\widehat P_h\underline V_{h+1}
-\tfrac12\beta_h^P\operatorname{span}(\underline V_{h+1}).
\]

The upper bound is symmetric.  Clipping preserves validity because all
remaining returns lie in \([0,H-h+1]\). \(\square\)

Combining Proposition 4 with Theorem 3 gives a fully explicit tabular offline
memory certificate.

## 6. Matching rates in a two-point family

### Construction

Use horizon two.  A cue \(B\in\{0,1\}\) is observed at stage one, with
\(\Pr(B=1)=p\leq1/2\), and the final current observation is constant.  The
behavior policy chooses each final action with probability \(1/2\).  For
\(0<\gamma\leq1/2\), set

\[
r_+=\frac12+\frac\gamma2,
\qquad
r_-=\frac12-\frac\gamma2.
\]

Define two environments:

- \(\mathcal M_0\): action zero has mean reward \(r_+\) and action one has
  mean reward \(r_-\), after either cue.
- \(\mathcal M_1\): the same holds after \(B=0\), but after the rare cue
  \(B=1\), the means are swapped.

Thus \(m_{\rm dec}(\mathcal M_0)=1\) and
\(m_{\rm dec}(\mathcal M_1)=2\).  In \(\mathcal M_1\), the one-memory
ambiguity is \(\gamma/2\).

### Theorem 4 (rare-context/action-gap lower bound)

For any estimator \(\widehat m\) based on \(n\) independent behavior-policy
episodes, if

\[
\max_{i\in\{0,1\}}
\Pr_{\mathcal M_i}(\widehat m\neq m_{\rm dec}(\mathcal M_i))\leq\delta
\]

for \(0<\delta<1/4\), then

\[
\boxed{
 n\geq \frac{\log(1/(4\delta))}{6p\gamma^2}.
}
\]

**Proof.** The models differ only after the rare cue.  Their one-episode KL
divergence is

\[
D_{\rm KL}(P_0\|P_1)
=\frac p2\left[
\operatorname{kl}(r_+\|r_-)
+\operatorname{kl}(r_-\|r_+)
\right].
\]

Using
\(\operatorname{kl}(u\|v)\leq(u-v)^2/[v(1-v)]\), and
\(r_+r_-=1/4-\gamma^2/4\geq3/16\), this is at most
\((16/3)p\gamma^2<6p\gamma^2\).  Bretagnolle--Huber gives

\[
\Pr_0(\widehat m\neq1)+\Pr_1(\widehat m\neq2)
\geq\tfrac12\exp[-D_{\rm KL}(P_0^n\|P_1^n)].
\]

If each error is at most \(\delta\), rearranging yields the result.
\(\square\)

### Corollary 2 (matching tabular upper rate)

Let \(M\) be the number of full-history state--action pairs.  Run the tabular
confidence construction with parameter \(\delta/2\) and DRMS with tolerance
\(\gamma/4\).  If

\[
n\geq \frac{32\log(8M/\delta)}{p\gamma^2},
\]

then DRMS identifies the correct memory in both models with probability at
least \(1-\delta\).

**Proof.** Each final state--action pair has probability at least \(p/2\).
A multiplicative Chernoff bound and union bound give

\[
\Pr\!\left(\min_{x,a}N_2(x,a)<np/4\right)
\leq4e^{-np/16}.
\]

The displayed sample size makes this at most \(\delta/2\).  On the simultaneous
reward-concentration event, every final reward radius satisfies

\[
\beta_2^r(x,a)
\leq\sqrt{\frac{2\log(8M/\delta)}{np}}
\leq\gamma/4.
\]

The corresponding lower bound of the best action exceeds the upper bound of
the other action in every final context.  Hence \(\overline G_1=0\) in
\(\mathcal M_0\), while in \(\mathcal M_1\),
\(\overline G_1\geq G_1=\gamma/2>\gamma/4\) and
\(\overline G_2=0\).  Thus DRMS selects the correct order. \(\square\)

The upper and lower rates match in \(p\) and \(\gamma\), up to logarithms and
constants.

## 7. What is proved versus deferred

Proved and implemented:

- population monotonicity and exact sufficiency;
- the value-loss sandwich;
- observation-prediction/decision separation constructions;
- simultaneous interval-based certificates;
- interval-tightness and memory-recovery conditions;
- valid tabular model-based intervals;
- matching two-point upper and lower rates in \(p\gamma^2\).

Deferred extensions:

- occupancy-weighted rather than uniform-over-context certificates;
- continuous observations or function approximation;
- selecting a representation jointly with memory;
- unknown or unbounded maximum full-context order;
- computationally compact alternatives to explicit history enumeration.
