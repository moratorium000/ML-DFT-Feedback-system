from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
from scipy.optimize import minimize

from interfaces import Structure, MutationResult, PathStep
from utils.logger import get_logger


@dataclass
class OptimizerParameters:
    """최적화 파라미터"""
    learning_rate: float = 0.01
    n_iterations: int = 100
    convergence_threshold: float = 1e-5
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_pressure: float = 0.7
    diversity_weight: float = 0.3


class MutationOptimizer:
    """Mutation 최적화기"""

    def __init__(self, params: Optional[OptimizerParameters] = None):
        self.params = params or OptimizerParameters()
        self.logger = get_logger(__name__)
        self.history = []

    async def optimize(self,
                       initial_mutations: List[MutationResult],
                       target_properties: Dict[str, float]) -> List[MutationResult]:
        """Mutation 최적화"""
        population = initial_mutations
        best_solution = None
        best_fitness = float('-inf')

        for iteration in range(self.params.n_iterations):
            # 적합도 평가
            fitness_scores = self._evaluate_fitness(population, target_properties)
            current_best_idx = np.argmax(fitness_scores)

            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_solution = population[current_best_idx]

            # 수렴 검사
            if self._check_convergence(fitness_scores):
                self.logger.info(f"Converged at iteration {iteration}")
                break

            # 다음 세대 생성
            population = self._create_next_generation(
                population,
                fitness_scores
            )

            # 히스토리 업데이트
            self.history.append({
                'iteration': iteration,
                'best_fitness': best_fitness,
                'mean_fitness': np.mean(fitness_scores),
                'diversity': self._calculate_diversity(population)
            })

        return self._select_best_mutations(population, target_properties)

    def _evaluate_fitness(self,
                          mutations: List[MutationResult],
                          target_properties: Dict[str, float]) -> np.ndarray:
        """적합도 평가"""
        fitness_scores = np.zeros(len(mutations))

        for i, mutation in enumerate(mutations):
            # 물성 일치도
            property_score = self._calculate_property_match(
                mutation,
                target_properties
            )

            # 안정성 점수
            stability_score = mutation.stability_score

            # 타당성 점수
            validity_score = mutation.validity_score

            # 종합 점수 계산
            fitness_scores[i] = (
                    0.4 * property_score +
                    0.3 * stability_score +
                    0.3 * validity_score
            )

        return fitness_scores

    def _create_next_generation(self,
                                population: List[MutationResult],
                                fitness_scores: np.ndarray) -> List[MutationResult]:
        """다음 세대 생성"""
        new_population = []

        # 엘리트 보존
        n_elite = int(0.1 * self.params.population_size)
        elite_idx = np.argsort(fitness_scores)[-n_elite:]
        new_population.extend([population[i] for i in elite_idx])

        while len(new_population) < self.params.population_size:
            if np.random.random() < self.params.crossover_rate:
                # 교차
                parent1 = self._select_parent(population, fitness_scores)
                parent2 = self._select_parent(population, fitness_scores)
                child = self._crossover(parent1, parent2)
            else:
                # 변이
                parent = self._select_parent(population, fitness_scores)
                child = self._mutate(parent)

            new_population.append(child)

        return new_population

    def _select_parent(self,
                       population: List[MutationResult],
                       fitness_scores: np.ndarray) -> MutationResult:
        """부모 선택 (토너먼트 선택)"""
        tournament_size = 3
        tournament_idx = np.random.choice(
            len(population),
            size=tournament_size,
            replace=False
        )
        tournament_fitness = fitness_scores[tournament_idx]
        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        return population[winner_idx]

    def _crossover(self,
                   parent1: MutationResult,
                   parent2: MutationResult) -> MutationResult:
        """교차 연산"""
        # 구조 교차
        crossover_point = np.random.randint(
            len(parent1.mutated_structure.positions)
        )

        child_structure = parent1.mutated_structure.copy()
        child_structure.positions[crossover_point:] = (
            parent2.mutated_structure.positions[crossover_point:]
        )

        # 새로운 MutationResult 생성
        return MutationResult(
            original_structure=parent1.original_structure,
            mutated_structure=child_structure,
            mutation_type="crossover",
            changes=self._describe_changes(
                parent1.original_structure,
                child_structure
            ),
            success=True,
            stability_score=0.0,  # 재계산 필요
            validity_score=0.0,  # 재계산 필요
            energy_estimate=0.0  # 재계산 필요
        )

    def _mutate(self, parent: MutationResult) -> MutationResult:
        """변이 연산"""
        mutated_structure = parent.mutated_structure.copy()

        # 무작위 원자 선택
        atom_idx = np.random.randint(len(mutated_structure.positions))

        # 변위 적용
        displacement = np.random.normal(0, 0.1, 3)  # 표준편차 0.1Å
        mutated_structure.positions[atom_idx] += displacement

        return MutationResult(
            original_structure=parent.original_structure,
            mutated_structure=mutated_structure,
            mutation_type="mutation",
            changes=self._describe_changes(
                parent.original_structure,
                mutated_structure
            ),
            success=True,
            stability_score=0.0,  # 재계산 필요
            validity_score=0.0,  # 재계산 필요
            energy_estimate=0.0  # 재계산 필요
        )

    def _calculate_diversity(self,
                             population: List[MutationResult]) -> float:
        """다양성 계산"""
        structures = [m.mutated_structure for m in population]
        n_structures = len(structures)

        if n_structures < 2:
            return 0.0

        # RMSD 기반 다양성
        diversity_sum = 0.0
        for i in range(n_structures):
            for j in range(i + 1, n_structures):
                rmsd = self._calculate_rmsd(
                    structures[i],
                    structures[j]
                )
                diversity_sum += rmsd

        return diversity_sum / (n_structures * (n_structures - 1) / 2)

    def _check_convergence(self, fitness_scores: np.ndarray) -> bool:
        """수렴 검사"""
        if len(self.history) < 2:
            return False

        prev_best = self.history[-1]['best_fitness']
        current_best = np.max(fitness_scores)

        return abs(current_best - prev_best) < self.params.convergence_threshold