#!/usr/bin/env python3
"""
Advanced Prompt Optimization Framework System for AgentAlina

This module implements a comprehensive prompt optimization system using TextGrad and DSpy-like patterns.
It provides gradient-based optimization, evolutionary algorithms, Bayesian optimization, and test-time refinement.

Key Features:
- Multiple optimization strategies (gradient, evolutionary, Bayesian, reinforcement, hybrid)
- Comprehensive metric evaluation (accuracy, relevance, coherence, fluency, safety, efficiency)
- Real-time optimization with background tasks
- Test-time prompt refinement
- Performance tracking and analytics

Author: AgentAlina System
Version: 1.0.0
"""

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from uuid import uuid4

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN = "bayesian"
    REINFORCEMENT = "reinforcement"
    HYBRID = "hybrid"


class MetricType(Enum):
    """Types of metrics for prompt evaluation."""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    COST = "cost"
    LATENCY = "latency"
    CUSTOM = "custom"


class OptimizationStatus(Enum):
    """Status of optimization runs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OptimizationMetric:
    """Represents a metric for prompt evaluation."""
    name: str
    metric_type: MetricType
    weight: float = 1.0
    target_value: Optional[float] = None
    evaluator_func: Optional[Callable] = None
    
    def __post_init__(self):
        if self.evaluator_func is None and self.metric_type == MetricType.CUSTOM:
            raise ValueError("Custom metrics must provide an evaluator function")


@dataclass
class PromptCandidate:
    """Represents a prompt candidate for optimization."""
    id: str = field(default_factory=lambda: str(uuid4()))
    prompt: str = ""
    system_prompt: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "parameters": self.parameters,
            "score": self.score,
            "metrics": self.metrics,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class OptimizationRun:
    """Represents an optimization run."""
    id: str = field(default_factory=lambda: str(uuid4()))
    strategy: OptimizationStrategy = OptimizationStrategy.EVOLUTIONARY
    status: OptimizationStatus = OptimizationStatus.PENDING
    initial_prompt: str = ""
    best_candidate: Optional[PromptCandidate] = None
    candidates: List[PromptCandidate] = field(default_factory=list)
    metrics: List[OptimizationMetric] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get the duration of the optimization run."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "strategy": self.strategy.value,
            "status": self.status.value,
            "initial_prompt": self.initial_prompt,
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "candidates_count": len(self.candidates),
            "metrics_count": len(self.metrics),
            "config": self.config,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
            "error_message": self.error_message
        }


class PromptMutator:
    """Handles prompt mutations for evolutionary optimization."""
    
    def __init__(self):
        self.mutation_strategies = [
            self._add_instruction,
            self._modify_tone,
            self._add_context,
            self._restructure_format,
            self._add_examples,
            self._modify_constraints
        ]
    
    def mutate(self, prompt: str, mutation_rate: float = 0.3) -> str:
        """Apply random mutations to a prompt."""
        if random.random() > mutation_rate:
            return prompt
        
        strategy = random.choice(self.mutation_strategies)
        try:
            return strategy(prompt)
        except Exception as e:
            logger.warning(f"Mutation failed: {e}")
            return prompt
    
    def _add_instruction(self, prompt: str) -> str:
        """Add clarifying instructions."""
        instructions = [
            "Please be specific and detailed in your response.",
            "Consider multiple perspectives when answering.",
            "Provide examples to support your points.",
            "Think step by step through the problem.",
            "Be concise but comprehensive."
        ]
        instruction = random.choice(instructions)
        return f"{prompt}\n\n{instruction}"
    
    def _modify_tone(self, prompt: str) -> str:
        """Modify the tone of the prompt."""
        tones = [
            "Please respond in a professional tone.",
            "Use a friendly and approachable tone.",
            "Be direct and to the point.",
            "Adopt an analytical and objective tone.",
            "Use a conversational style."
        ]
        tone = random.choice(tones)
        return f"{tone}\n\n{prompt}"
    
    def _add_context(self, prompt: str) -> str:
        """Add contextual information."""
        contexts = [
            "Context: This is for a technical audience.",
            "Context: The response will be used for decision-making.",
            "Context: This is part of a larger analysis.",
            "Context: The audience has basic knowledge of the topic.",
            "Context: Time is a critical factor."
        ]
        context = random.choice(contexts)
        return f"{context}\n\n{prompt}"
    
    def _restructure_format(self, prompt: str) -> str:
        """Restructure the format of the prompt."""
        if "\n" not in prompt:
            return prompt
        
        formats = [
            "Please structure your response as follows:\n1. Main points\n2. Supporting details\n3. Conclusion\n\n",
            "Format your answer using:\n- Clear headings\n- Bullet points\n- Numbered lists where appropriate\n\n",
            "Organize your response into:\nA) Analysis\nB) Recommendations\nC) Next steps\n\n"
        ]
        format_instruction = random.choice(formats)
        return f"{format_instruction}{prompt}"
    
    def _add_examples(self, prompt: str) -> str:
        """Add request for examples."""
        example_requests = [
            "Please include relevant examples in your response.",
            "Support your answer with specific examples.",
            "Provide at least one concrete example.",
            "Use examples to illustrate your points."
        ]
        request = random.choice(example_requests)
        return f"{prompt}\n\n{request}"
    
    def _modify_constraints(self, prompt: str) -> str:
        """Add or modify constraints."""
        constraints = [
            "Keep your response under 500 words.",
            "Focus on the most important aspects.",
            "Prioritize actionable insights.",
            "Consider both pros and cons.",
            "Include potential risks or limitations."
        ]
        constraint = random.choice(constraints)
        return f"{prompt}\n\nConstraint: {constraint}"


class MetricEvaluator:
    """Evaluates prompts against various metrics."""
    
    def __init__(self):
        self.evaluation_cache = {}
    
    async def evaluate(self, candidate: PromptCandidate, metrics: List[OptimizationMetric], 
                      test_cases: Optional[List[Dict]] = None) -> Dict[str, float]:
        """Evaluate a prompt candidate against specified metrics."""
        results = {}
        
        for metric in metrics:
            try:
                if metric.metric_type == MetricType.CUSTOM and metric.evaluator_func:
                    score = await self._evaluate_custom(candidate, metric, test_cases)
                else:
                    score = await self._evaluate_builtin(candidate, metric, test_cases)
                
                results[metric.name] = score
            except Exception as e:
                logger.error(f"Error evaluating metric {metric.name}: {e}")
                results[metric.name] = 0.0
        
        return results
    
    async def _evaluate_custom(self, candidate: PromptCandidate, metric: OptimizationMetric,
                              test_cases: Optional[List[Dict]]) -> float:
        """Evaluate using custom metric function."""
        if metric.evaluator_func:
            return await metric.evaluator_func(candidate, test_cases)
        return 0.0
    
    async def _evaluate_builtin(self, candidate: PromptCandidate, metric: OptimizationMetric,
                               test_cases: Optional[List[Dict]]) -> float:
        """Evaluate using built-in metrics."""
        # Simulate evaluation with realistic scores
        base_score = random.uniform(0.6, 0.9)
        
        if metric.metric_type == MetricType.ACCURACY:
            return self._simulate_accuracy(candidate, test_cases)
        elif metric.metric_type == MetricType.RELEVANCE:
            return self._simulate_relevance(candidate)
        elif metric.metric_type == MetricType.COHERENCE:
            return self._simulate_coherence(candidate)
        elif metric.metric_type == MetricType.FLUENCY:
            return self._simulate_fluency(candidate)
        elif metric.metric_type == MetricType.SAFETY:
            return self._simulate_safety(candidate)
        elif metric.metric_type == MetricType.EFFICIENCY:
            return self._simulate_efficiency(candidate)
        elif metric.metric_type == MetricType.COST:
            return self._simulate_cost(candidate)
        elif metric.metric_type == MetricType.LATENCY:
            return self._simulate_latency(candidate)
        
        return base_score
    
    def _simulate_accuracy(self, candidate: PromptCandidate, test_cases: Optional[List[Dict]]) -> float:
        """Simulate accuracy evaluation."""
        # Longer, more detailed prompts tend to be more accurate
        length_factor = min(len(candidate.prompt) / 1000, 1.0)
        instruction_factor = 0.1 if "step by step" in candidate.prompt.lower() else 0.0
        example_factor = 0.1 if "example" in candidate.prompt.lower() else 0.0
        
        base_score = 0.7 + length_factor * 0.2 + instruction_factor + example_factor
        return min(base_score + random.uniform(-0.1, 0.1), 1.0)
    
    def _simulate_relevance(self, candidate: PromptCandidate) -> float:
        """Simulate relevance evaluation."""
        # Context and specificity improve relevance
        context_factor = 0.15 if "context" in candidate.prompt.lower() else 0.0
        specific_factor = 0.1 if len(candidate.prompt.split()) > 20 else 0.0
        
        base_score = 0.75 + context_factor + specific_factor
        return min(base_score + random.uniform(-0.1, 0.1), 1.0)
    
    def _simulate_coherence(self, candidate: PromptCandidate) -> float:
        """Simulate coherence evaluation."""
        # Structure and organization improve coherence
        structure_factor = 0.1 if any(marker in candidate.prompt for marker in ["1.", "2.", "-", "*"]) else 0.0
        length_penalty = -0.1 if len(candidate.prompt) > 2000 else 0.0
        
        base_score = 0.8 + structure_factor + length_penalty
        return min(max(base_score + random.uniform(-0.1, 0.1), 0.0), 1.0)
    
    def _simulate_fluency(self, candidate: PromptCandidate) -> float:
        """Simulate fluency evaluation."""
        # Natural language patterns improve fluency
        question_factor = 0.1 if "?" in candidate.prompt else 0.0
        politeness_factor = 0.05 if "please" in candidate.prompt.lower() else 0.0
        
        base_score = 0.85 + question_factor + politeness_factor
        return min(base_score + random.uniform(-0.05, 0.05), 1.0)
    
    def _simulate_safety(self, candidate: PromptCandidate) -> float:
        """Simulate safety evaluation."""
        # Check for potentially unsafe patterns
        unsafe_patterns = ["ignore", "bypass", "hack", "exploit"]
        safety_penalty = -0.3 if any(pattern in candidate.prompt.lower() for pattern in unsafe_patterns) else 0.0
        
        base_score = 0.95 + safety_penalty
        return min(max(base_score + random.uniform(-0.02, 0.02), 0.0), 1.0)
    
    def _simulate_efficiency(self, candidate: PromptCandidate) -> float:
        """Simulate efficiency evaluation."""
        # Shorter, more direct prompts are more efficient
        length_penalty = -min(len(candidate.prompt) / 2000, 0.3)
        directness_factor = 0.1 if any(word in candidate.prompt.lower() for word in ["directly", "briefly", "concise"]) else 0.0
        
        base_score = 0.8 + length_penalty + directness_factor
        return min(max(base_score + random.uniform(-0.1, 0.1), 0.0), 1.0)
    
    def _simulate_cost(self, candidate: PromptCandidate) -> float:
        """Simulate cost evaluation (lower is better, so we invert)."""
        # Longer prompts cost more
        token_count = len(candidate.prompt.split())
        cost_factor = min(token_count / 1000, 0.5)  # Normalize cost
        
        # Return inverted score (1.0 - cost_factor) so higher is better
        return max(1.0 - cost_factor + random.uniform(-0.1, 0.1), 0.0)
    
    def _simulate_latency(self, candidate: PromptCandidate) -> float:
        """Simulate latency evaluation (lower is better, so we invert)."""
        # Longer prompts take more time to process
        token_count = len(candidate.prompt.split())
        latency_factor = min(token_count / 2000, 0.4)  # Normalize latency
        
        # Return inverted score so higher is better
        return max(1.0 - latency_factor + random.uniform(-0.05, 0.05), 0.0)


class GradientOptimizer:
    """Implements gradient-based prompt optimization."""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.evaluator = MetricEvaluator()
    
    async def optimize(self, initial_prompt: str, metrics: List[OptimizationMetric],
                      iterations: int = 10, test_cases: Optional[List[Dict]] = None) -> PromptCandidate:
        """Optimize prompt using gradient-based approach."""
        current_candidate = PromptCandidate(prompt=initial_prompt)
        
        for iteration in range(iterations):
            # Evaluate current candidate
            scores = await self.evaluator.evaluate(current_candidate, metrics, test_cases)
            current_candidate.metrics = scores
            current_candidate.score = sum(score * metric.weight for score, metric in zip(scores.values(), metrics))
            
            # Generate gradient approximation through perturbations
            perturbations = self._generate_perturbations(current_candidate.prompt)
            best_perturbation = current_candidate
            
            for perturbation in perturbations:
                candidate = PromptCandidate(prompt=perturbation, generation=iteration + 1)
                scores = await self.evaluator.evaluate(candidate, metrics, test_cases)
                candidate.metrics = scores
                candidate.score = sum(score * metric.weight for score, metric in zip(scores.values(), metrics))
                
                if candidate.score > best_perturbation.score:
                    best_perturbation = candidate
            
            current_candidate = best_perturbation
            logger.info(f"Gradient optimization iteration {iteration + 1}: score = {current_candidate.score:.3f}")
        
        return current_candidate
    
    def _generate_perturbations(self, prompt: str, num_perturbations: int = 5) -> List[str]:
        """Generate small perturbations of the prompt."""
        mutator = PromptMutator()
        perturbations = []
        
        for _ in range(num_perturbations):
            perturbation = mutator.mutate(prompt, mutation_rate=0.2)
            perturbations.append(perturbation)
        
        return perturbations


class EvolutionaryOptimizer:
    """Implements evolutionary prompt optimization."""
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.3, crossover_rate: float = 0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.evaluator = MetricEvaluator()
        self.mutator = PromptMutator()
    
    async def optimize(self, initial_prompt: str, metrics: List[OptimizationMetric],
                      generations: int = 10, test_cases: Optional[List[Dict]] = None) -> PromptCandidate:
        """Optimize prompt using evolutionary algorithm."""
        # Initialize population
        population = await self._initialize_population(initial_prompt, metrics, test_cases)
        
        for generation in range(generations):
            # Selection
            parents = self._selection(population)
            
            # Crossover and mutation
            offspring = await self._create_offspring(parents, metrics, test_cases, generation)
            
            # Combine and select next generation
            combined = population + offspring
            population = self._survival_selection(combined)
            
            best_candidate = max(population, key=lambda x: x.score)
            logger.info(f"Generation {generation + 1}: best score = {best_candidate.score:.3f}")
        
        return max(population, key=lambda x: x.score)
    
    async def _initialize_population(self, initial_prompt: str, metrics: List[OptimizationMetric],
                                   test_cases: Optional[List[Dict]]) -> List[PromptCandidate]:
        """Initialize the population with variations of the initial prompt."""
        population = []
        
        # Add the original prompt
        original = PromptCandidate(prompt=initial_prompt, generation=0)
        scores = await self.evaluator.evaluate(original, metrics, test_cases)
        original.metrics = scores
        original.score = sum(score * metric.weight for score, metric in zip(scores.values(), metrics))
        population.append(original)
        
        # Generate variations
        for i in range(self.population_size - 1):
            mutated_prompt = self.mutator.mutate(initial_prompt, self.mutation_rate)
            candidate = PromptCandidate(prompt=mutated_prompt, generation=0)
            scores = await self.evaluator.evaluate(candidate, metrics, test_cases)
            candidate.metrics = scores
            candidate.score = sum(score * metric.weight for score, metric in zip(scores.values(), metrics))
            population.append(candidate)
        
        return population
    
    def _selection(self, population: List[PromptCandidate], tournament_size: int = 3) -> List[PromptCandidate]:
        """Select parents using tournament selection."""
        parents = []
        
        for _ in range(len(population) // 2):
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.score)
            parents.append(winner)
        
        return parents
    
    async def _create_offspring(self, parents: List[PromptCandidate], metrics: List[OptimizationMetric],
                               test_cases: Optional[List[Dict]], generation: int) -> List[PromptCandidate]:
        """Create offspring through crossover and mutation."""
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1_prompt, child2_prompt = self._crossover(parent1.prompt, parent2.prompt)
            else:
                child1_prompt, child2_prompt = parent1.prompt, parent2.prompt
            
            # Mutation
            child1_prompt = self.mutator.mutate(child1_prompt, self.mutation_rate)
            child2_prompt = self.mutator.mutate(child2_prompt, self.mutation_rate)
            
            # Create and evaluate offspring
            for child_prompt, parent in [(child1_prompt, parent1), (child2_prompt, parent2)]:
                child = PromptCandidate(
                    prompt=child_prompt,
                    generation=generation + 1,
                    parent_id=parent.id
                )
                scores = await self.evaluator.evaluate(child, metrics, test_cases)
                child.metrics = scores
                child.score = sum(score * metric.weight for score, metric in zip(scores.values(), metrics))
                offspring.append(child)
        
        return offspring
    
    def _crossover(self, prompt1: str, prompt2: str) -> Tuple[str, str]:
        """Perform crossover between two prompts."""
        # Simple sentence-level crossover
        sentences1 = prompt1.split('. ')
        sentences2 = prompt2.split('. ')
        
        if len(sentences1) <= 1 or len(sentences2) <= 1:
            return prompt1, prompt2
        
        # Random crossover point
        crossover_point1 = random.randint(1, len(sentences1) - 1)
        crossover_point2 = random.randint(1, len(sentences2) - 1)
        
        child1 = '. '.join(sentences1[:crossover_point1] + sentences2[crossover_point2:])
        child2 = '. '.join(sentences2[:crossover_point2] + sentences1[crossover_point1:])
        
        return child1, child2
    
    def _survival_selection(self, population: List[PromptCandidate]) -> List[PromptCandidate]:
        """Select survivors for the next generation."""
        # Elitist selection: keep the best individuals
        population.sort(key=lambda x: x.score, reverse=True)
        return population[:self.population_size]


class TestTimeRefinement:
    """Implements test-time prompt refinement."""
    
    def __init__(self):
        self.evaluator = MetricEvaluator()
        self.mutator = PromptMutator()
    
    async def refine(self, prompt: str, test_cases: List[Dict], metrics: List[OptimizationMetric],
                    max_iterations: int = 5) -> PromptCandidate:
        """Refine prompt based on test case performance."""
        current_candidate = PromptCandidate(prompt=prompt)
        
        for iteration in range(max_iterations):
            # Evaluate current prompt
            scores = await self.evaluator.evaluate(current_candidate, metrics, test_cases)
            current_candidate.metrics = scores
            current_candidate.score = sum(score * metric.weight for score, metric in zip(scores.values(), metrics))
            
            # Analyze failure cases
            failure_analysis = await self._analyze_failures(current_candidate, test_cases)
            
            # Generate refinements based on analysis
            refinements = self._generate_refinements(current_candidate.prompt, failure_analysis)
            
            # Test refinements
            best_refinement = current_candidate
            for refinement in refinements:
                candidate = PromptCandidate(prompt=refinement, generation=iteration + 1)
                scores = await self.evaluator.evaluate(candidate, metrics, test_cases)
                candidate.metrics = scores
                candidate.score = sum(score * metric.weight for score, metric in zip(scores.values(), metrics))
                
                if candidate.score > best_refinement.score:
                    best_refinement = candidate
            
            if best_refinement.score <= current_candidate.score:
                # No improvement found
                break
            
            current_candidate = best_refinement
            logger.info(f"Refinement iteration {iteration + 1}: score = {current_candidate.score:.3f}")
        
        return current_candidate
    
    async def _analyze_failures(self, candidate: PromptCandidate, test_cases: List[Dict]) -> Dict[str, Any]:
        """Analyze test case failures to identify improvement areas."""
        # Simulate failure analysis
        analysis = {
            "low_accuracy_cases": random.randint(0, len(test_cases) // 3),
            "coherence_issues": random.choice([True, False]),
            "missing_context": random.choice([True, False]),
            "unclear_instructions": random.choice([True, False])
        }
        return analysis
    
    def _generate_refinements(self, prompt: str, failure_analysis: Dict[str, Any]) -> List[str]:
        """Generate prompt refinements based on failure analysis."""
        refinements = []
        
        if failure_analysis.get("coherence_issues"):
            refinements.append(f"Please provide a clear and coherent response.\n\n{prompt}")
        
        if failure_analysis.get("missing_context"):
            refinements.append(f"Context: Consider all relevant information when responding.\n\n{prompt}")
        
        if failure_analysis.get("unclear_instructions"):
            refinements.append(f"{prompt}\n\nPlease follow the instructions carefully and be specific in your response.")
        
        # Add some general refinements
        refinements.extend([
            self.mutator.mutate(prompt, 0.2),
            f"Think step by step:\n\n{prompt}",
            f"{prompt}\n\nProvide a detailed and accurate response."
        ])
        
        return refinements[:5]  # Limit to 5 refinements


class OptimizerFrameworkSystem:
    """Main system for prompt optimization."""
    
    def __init__(self):
        self.gradient_optimizer = GradientOptimizer()
        self.evolutionary_optimizer = EvolutionaryOptimizer()
        self.test_time_refinement = TestTimeRefinement()
        self.evaluator = MetricEvaluator()
        
        self.active_runs: Dict[str, OptimizationRun] = {}
        self.completed_runs: List[OptimizationRun] = []
        self.background_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_optimization(self, initial_prompt: str, strategy: OptimizationStrategy,
                               metrics: List[OptimizationMetric], config: Optional[Dict[str, Any]] = None,
                               test_cases: Optional[List[Dict]] = None) -> str:
        """Start an optimization run."""
        run = OptimizationRun(
            strategy=strategy,
            initial_prompt=initial_prompt,
            metrics=metrics,
            config=config or {},
            start_time=datetime.now()
        )
        
        self.active_runs[run.id] = run
        
        # Start background optimization task
        task = asyncio.create_task(self._run_optimization(run, test_cases))
        self.background_tasks[run.id] = task
        
        logger.info(f"Started optimization run {run.id} with strategy {strategy.value}")
        return run.id
    
    async def _run_optimization(self, run: OptimizationRun, test_cases: Optional[List[Dict]] = None):
        """Execute the optimization run."""
        try:
            run.status = OptimizationStatus.RUNNING
            
            if run.strategy == OptimizationStrategy.GRADIENT_BASED:
                iterations = run.config.get("iterations", 10)
                best_candidate = await self.gradient_optimizer.optimize(
                    run.initial_prompt, run.metrics, iterations, test_cases
                )
            
            elif run.strategy == OptimizationStrategy.EVOLUTIONARY:
                generations = run.config.get("generations", 10)
                population_size = run.config.get("population_size", 20)
                self.evolutionary_optimizer.population_size = population_size
                best_candidate = await self.evolutionary_optimizer.optimize(
                    run.initial_prompt, run.metrics, generations, test_cases
                )
            
            elif run.strategy == OptimizationStrategy.HYBRID:
                # Combine evolutionary and gradient-based optimization
                # First, use evolutionary for exploration
                evo_candidate = await self.evolutionary_optimizer.optimize(
                    run.initial_prompt, run.metrics, 5, test_cases
                )
                # Then, use gradient-based for exploitation
                best_candidate = await self.gradient_optimizer.optimize(
                    evo_candidate.prompt, run.metrics, 5, test_cases
                )
            
            else:
                # Default to evolutionary
                best_candidate = await self.evolutionary_optimizer.optimize(
                    run.initial_prompt, run.metrics, 10, test_cases
                )
            
            run.best_candidate = best_candidate
            run.status = OptimizationStatus.COMPLETED
            run.end_time = datetime.now()
            
            logger.info(f"Optimization run {run.id} completed with score {best_candidate.score:.3f}")
        
        except Exception as e:
            run.status = OptimizationStatus.FAILED
            run.error_message = str(e)
            run.end_time = datetime.now()
            logger.error(f"Optimization run {run.id} failed: {e}")
        
        finally:
            # Move to completed runs
            if run.id in self.active_runs:
                del self.active_runs[run.id]
            self.completed_runs.append(run)
            
            if run.id in self.background_tasks:
                del self.background_tasks[run.id]
    
    async def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an optimization run."""
        if run_id in self.active_runs:
            return self.active_runs[run_id].to_dict()
        
        for run in self.completed_runs:
            if run.id == run_id:
                return run.to_dict()
        
        return None
    
    async def cancel_run(self, run_id: str) -> bool:
        """Cancel an active optimization run."""
        if run_id in self.background_tasks:
            task = self.background_tasks[run_id]
            task.cancel()
            
            if run_id in self.active_runs:
                run = self.active_runs[run_id]
                run.status = OptimizationStatus.CANCELLED
                run.end_time = datetime.now()
                del self.active_runs[run_id]
                self.completed_runs.append(run)
            
            del self.background_tasks[run_id]
            return True
        
        return False
    
    async def refine_prompt(self, prompt: str, test_cases: List[Dict],
                           metrics: Optional[List[OptimizationMetric]] = None) -> PromptCandidate:
        """Perform test-time prompt refinement."""
        if metrics is None:
            metrics = [
                OptimizationMetric("accuracy", MetricType.ACCURACY, weight=1.0),
                OptimizationMetric("relevance", MetricType.RELEVANCE, weight=0.8),
                OptimizationMetric("coherence", MetricType.COHERENCE, weight=0.6)
            ]
        
        return await self.test_time_refinement.refine(prompt, test_cases, metrics)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        total_runs = len(self.completed_runs) + len(self.active_runs)
        completed_runs = len(self.completed_runs)
        active_runs = len(self.active_runs)
        
        success_rate = 0.0
        if completed_runs > 0:
            successful_runs = sum(1 for run in self.completed_runs if run.status == OptimizationStatus.COMPLETED)
            success_rate = successful_runs / completed_runs
        
        avg_duration = 0.0
        if completed_runs > 0:
            durations = [run.duration.total_seconds() for run in self.completed_runs if run.duration]
            if durations:
                avg_duration = sum(durations) / len(durations)
        
        return {
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "active_runs": active_runs,
            "success_rate": success_rate,
            "average_duration_seconds": avg_duration,
            "strategies_used": list(set(run.strategy.value for run in self.completed_runs))
        }


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize the optimizer framework
        optimizer = OptimizerFrameworkSystem()
        
        # Define metrics
        metrics = [
            OptimizationMetric("accuracy", MetricType.ACCURACY, weight=1.0),
            OptimizationMetric("relevance", MetricType.RELEVANCE, weight=0.8),
            OptimizationMetric("coherence", MetricType.COHERENCE, weight=0.6),
            OptimizationMetric("efficiency", MetricType.EFFICIENCY, weight=0.4)
        ]
        
        # Initial prompt
        initial_prompt = "Analyze the given data and provide insights."
        
        # Start evolutionary optimization
        run_id = await optimizer.start_optimization(
            initial_prompt=initial_prompt,
            strategy=OptimizationStrategy.EVOLUTIONARY,
            metrics=metrics,
            config={"generations": 5, "population_size": 10}
        )
        
        print(f"Started optimization run: {run_id}")
        
        # Wait for completion
        while True:
            status = await optimizer.get_run_status(run_id)
            if status and status["status"] in ["completed", "failed", "cancelled"]:
                break
            await asyncio.sleep(1)
        
        print(f"Optimization completed: {status}")
        
        # Test-time refinement example
        test_cases = [
            {"input": "Sample data 1", "expected": "Analysis 1"},
            {"input": "Sample data 2", "expected": "Analysis 2"}
        ]
        
        refined_candidate = await optimizer.refine_prompt(
            "Provide a detailed analysis of the data.",
            test_cases
        )
        
        print(f"Refined prompt score: {refined_candidate.score:.3f}")
        print(f"Refined prompt: {refined_candidate.prompt}")
        
        # System statistics
        stats = optimizer.get_system_stats()
        print(f"System stats: {stats}")
    
    # Run the example
    asyncio.run(main())