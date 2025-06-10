---
title: Database Management System
---

**Topics**

1. Loseless Decomposition
2. Dependency Preservation
3. Functional Dependency
4. Closure
5. First Normal Form
6. Second Normal Form
7. Third Normal Form
8. Trivial Dependency
9. Transitive Dependency
10. Super Key
11. Candidate Key
12. Primary Key
13. Boyce-Codd Normal Form

---

**Redundancy and Anomaly**
Having multiple copies of the same data in database is called redundancy, and database operations on such databases cause anomalies.

**Normalization** --> _Good Decomposition_ --> Minimization of Dependency

---

### Functional Dependency

dept_name --> Budget, Building

### Lossy Decomposition

When 1 of the 2 decomposed relations is a weak entity set, then we may have lossy decomposition. Example:

```txt
Suppose we decompose
employee(ID, name, street, city, salary) into
employee1 (ID, name)
employee2 (name, street, city, salary)
• Note that if name can be duplicate, then employee2 is a weak entity set and cannot
exist without an identifying relationship
• Consequently, this decomposition cannot preserve the information
```

### Loseless-Join Decomposition

Is a decomposition of a Relation _R_ into relations $R_1$, $R_2$, such that if we perform natural join of 2 smaller relations it will return the original relation

Decomposition is lossy if R1 cartestion join R2 ⊃ R
Decomposition is lossless if R1 cartesian join R2 = R

### 1NF

```
◦ the domains of all attributes of R are atomic
◦ the value of each attribute contains only a single value from that domain
```

### Functional Dependency

```
Let R be a relation schema
α ⊆ R and β ⊆ R
• The functional dependency or FD
α → β
holds on R if and only if for any legal relations r(R), whenever any two tuples t1 and t2
of r agree on the attributes α, they also agree on the attributes β. That is,

t1[α] = t2[α] ⇒ t1[β] = t2[β]
```

_We say that F holds on R if all legal relations on R satisfy the set of functional
dependencies F_

### Super Key

`K` is a super-key if and only if `k` --> `R`

### Candidate Key

`K` is a super-key and there exists no $\alpha$ subset of `K`, and $\alpha$ --> `R`

### Trivial Dependency

$\alpha \rightarrow \beta$ , where $\alpha \subseteq \beta$

### Closure of FDs

$F = {A \rightarrow B, B \rightarrow C}$
$F = {A \rightarrow B, B \rightarrow C, AB \rightarrow C}$

### Armstrong's Axioms

Reflexivity,
Augmentation,
Transitivity

**Derived Rules**
_Union_: alpha -> beta, alpha -> gamma, alpha -> beta gamma
_Decomposition_: Opposite of Union
Pseudotransitivity: alpha -> beta, gamma beta -> delta, then alpha gamma -> delta

### Closure of Attribute Sets

Given a an attribute set, using the FDs, we build up the attribute set to its further extent possible.

**Use cases of closure algorithm for attribute sets**

1. Testing for Superkey
2. Testing for Functional Dependency
3. Computing Closure of F

### BCNF

A relation schema is set to be BCNF with respect to a set F of FDs, if all the functional dependencies ($\alpha \rightarrow \beta$) in the closure set of F+ are either trivial or $\alpha$ is a superkey

#### Decomposition

if a non-trivial dependency causes violtion of BCNF, decompose to $\alpha$ union $\beta$ and (R - beta + alpha)

### Algorithm to Check for Loseless Decomposition

Union of decomposed relations must be the original relation
Intersection of decomposed relations mustn't be empty
Intersection of decomposed relations must be the key for atleast 1 relation

### Dependency Preservation

If we can verify a functional dependency by only check it on 1 relation, instead of having to join the relations, then such a decomposition is dependency preserving.

BCNF is not dependency preservation.
3NF is.

### 3NF

alpha -> beta in F+,
either
alpha -> beta is trivial,
alpha is superkey
Each attribute in beta - alpha is contained in a candidate key. (each attribute may be in a different candidate key)

### 2NF

Is 1NF
Does not contain partial dependency

### Partial Dependency

Y --> A, where A is a non-prime attribute(prime attribute is an attribute in the candidate key of the relation), and Y is a proper subset of a candidate key

### 3NF again

Is in 2NF,
Should not contain transitive dependency
X ->A, where A is part of some candidate key

### Transitive Dependency

Transitive dependency occurred because a non-key attribute (Author) was determining
another non-key attribute (Author Nationality).

```
A transitive dependency is a functional dependency which holds by virtue of
transitivity. A transitive dependency can occur only in a relation that has three or more
attributes.
• Let A, B, and C designate three distinct attributes (or distinct collections of attributes)
in the relation. Suppose all three of the following conditions hold:
◦ A → B
◦ It is not the case that B → A
◦ B → C
• Then the functional dependency A → C (which follows from 1 and 3 by the axiom of
transitivity) is a transitive dependency
```

### Testing for 3NF

### Algorithm for 3NF Decomposition

### Testing for BCNF

### Testing for BCNF Decomposition
