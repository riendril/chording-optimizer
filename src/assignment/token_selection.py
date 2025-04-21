"""
Token selection module for chord optimization.

This module provides algorithms for selecting tokens for chord assignment
using both greedy and genetic approaches.
"""

import copy
import logging
import random
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.common.config import GeneratorConfig
from src.common.shared_types import ChordData, TokenCollection, TokenData

logger = logging.getLogger(__name__)

# -----------------
# Mock Chord Creation
# -----------------


def create_mock_chord(token: str) -> ChordData:
    """
    Create a mock chord with difficulty 1 for prototyping.

    Args:
        token: Token to create chord for

    Returns:
        ChordData with difficulty 1
    """
    # Use first few letters as the chord
    letters = token[: min(3, len(token))]

    # Create a chord with minimal information and fixed difficulty
    return ChordData(letters=letters, keys=tuple())  # Empty for now


# -----------------
# Greedy Algorithm
# -----------------


def greedy_token_selection(
    token_collection: TokenCollection,
    max_tokens: int,
    recalculate_scores: Callable[[Set[str]], Dict[str, float]],
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> TokenCollection:
    """
    Greedy algorithm for selecting tokens:
    1. Pick the highest-scoring token
    2. Recalculate scores based on context
    3. Repeat until max_tokens are selected

    Args:
        token_collection: Collection of tokens with scores
        max_tokens: Maximum number of tokens to select
        recalculate_scores: Function to recalculate scores based on selections
        progress_callback: Optional callback for progress updates

    Returns:
        TokenCollection with assigned chords
    """
    # Create a working copy
    working_collection = copy.deepcopy(token_collection)

    # Keep track of selected tokens
    selected_tokens = set()

    # Track selection progress
    selection_count = 0

    while selection_count < max_tokens:
        # Sort tokens by score (descending)
        working_collection.tokens.sort(key=lambda t: t.score, reverse=True)

        # Find highest scoring unassigned token
        best_token = None
        for token in working_collection.tokens:
            if not hasattr(token, "assigned_chord") or token.assigned_chord is None:
                best_token = token
                break

        # If no valid token found, break
        if not best_token:
            logger.info(
                f"No more unassigned tokens available after {selection_count} selections"
            )
            break

        # Assign a chord to this token
        best_token.assigned_chord = create_mock_chord(best_token.original)
        selected_tokens.add(best_token.original)
        selection_count += 1

        # Report progress
        if progress_callback and selection_count % 10 == 0:
            progress_callback(
                selection_count, f"Selected {selection_count}/{max_tokens} tokens"
            )

        # Recalculate scores for remaining tokens
        new_scores = recalculate_scores(selected_tokens)

        # Update scores in the collection
        for token in working_collection.tokens:
            if token.original in new_scores:
                token.score = new_scores[token.original]

    logger.info(f"Greedy selection completed with {selection_count} tokens")
    return working_collection


# -----------------
# Genetic Algorithm
# -----------------


class TokenSelectionGenome:
    """Represents a genome for the genetic algorithm"""

    def __init__(
        self,
        token_collection: TokenCollection,
        selection_size: int,
        selected_tokens: Optional[Set[str]] = None,
        random_init: bool = False,
    ):
        """
        Initialize a genome with selected tokens.

        Args:
            token_collection: Token collection to optimize
            selection_size: Number of tokens to select
            selected_tokens: Pre-selected tokens (optional)
            random_init: Whether to initialize randomly
        """
        self.token_collection = token_collection
        self.selection_size = min(selection_size, len(token_collection.tokens))

        # Get all available tokens
        self.all_tokens = [t.original for t in token_collection.tokens]

        # Initialize selected tokens
        if selected_tokens:
            self.selected_tokens = set(selected_tokens)
        elif random_init:
            self.selected_tokens = set(
                random.sample(self.all_tokens, self.selection_size)
            )
        else:
            # Default to highest scoring tokens
            sorted_tokens = sorted(
                token_collection.tokens, key=lambda t: t.score, reverse=True
            )
            self.selected_tokens = set(
                t.original for t in sorted_tokens[: self.selection_size]
            )

        # Fitness is initially unset
        self.fitness = 0.0

    def calculate_fitness(
        self, recalculate_scores: Callable[[Set[str]], Dict[str, float]]
    ) -> float:
        """
        Calculate fitness as the total score of the token set.

        Args:
            recalculate_scores: Function to get scores for unselected tokens

        Returns:
            Fitness score (higher is better)
        """
        # Get scores for unselected tokens
        new_scores = recalculate_scores(self.selected_tokens)

        # For selected tokens, use their original score as value
        token_map = {t.original: t for t in self.token_collection.tokens}

        fitness = 0.0

        # Add score for selected tokens
        for token in self.selected_tokens:
            token_data = token_map.get(token)
            if token_data:
                fitness += token_data.score

        # Add scores for unselected tokens
        for token, score in new_scores.items():
            if token not in self.selected_tokens:
                fitness += score

        self.fitness = fitness
        return fitness

    def mutate(self, mutation_rate: float = 0.1) -> None:
        """
        Mutate by randomly replacing tokens.

        Args:
            mutation_rate: Probability of mutation for each token
        """
        # Get unselected tokens
        available_tokens = [t for t in self.all_tokens if t not in self.selected_tokens]

        if not available_tokens:
            return

        # Convert set to list for indexed access
        selected_list = list(self.selected_tokens)

        # Potentially mutate each token
        for i in range(len(selected_list)):
            if random.random() < mutation_rate:
                # Remove this token
                token_to_replace = selected_list[i]
                self.selected_tokens.remove(token_to_replace)

                # Add a random new token
                new_token = random.choice(available_tokens)
                self.selected_tokens.add(new_token)

                # Update available tokens
                available_tokens.remove(new_token)
                available_tokens.append(token_to_replace)

    def crossover(
        self, other: "TokenSelectionGenome"
    ) -> Tuple["TokenSelectionGenome", "TokenSelectionGenome"]:
        """
        Perform crossover with another genome.

        Args:
            other: Another genome to cross with

        Returns:
            Two new genomes resulting from crossover
        """
        # Convert to lists for indexed operations
        self_tokens = list(self.selected_tokens)
        other_tokens = list(other.selected_tokens)

        # Choose a random crossover point
        crossover_point = random.randint(
            1, min(len(self_tokens), len(other_tokens)) - 1
        )

        # Create new token selections
        child1_tokens = set(
            self_tokens[:crossover_point] + other_tokens[crossover_point:]
        )
        child2_tokens = set(
            other_tokens[:crossover_point] + self_tokens[crossover_point:]
        )

        # Ensure we have the correct number of tokens
        while len(child1_tokens) < self.selection_size:
            available = [t for t in self.all_tokens if t not in child1_tokens]
            if not available:
                break
            child1_tokens.add(random.choice(available))

        while len(child2_tokens) < self.selection_size:
            available = [t for t in self.all_tokens if t not in child2_tokens]
            if not available:
                break
            child2_tokens.add(random.choice(available))

        # Create new genomes
        child1 = TokenSelectionGenome(
            self.token_collection, self.selection_size, child1_tokens
        )

        child2 = TokenSelectionGenome(
            self.token_collection, self.selection_size, child2_tokens
        )

        return child1, child2


def tournament_selection(
    population: List[TokenSelectionGenome], tournament_size: int
) -> TokenSelectionGenome:
    """
    Select a genome using tournament selection.

    Args:
        population: List of genomes
        tournament_size: Number of genomes to include in tournament

    Returns:
        Selected genome
    """
    # Randomly select tournament_size individuals
    tournament = random.sample(population, min(tournament_size, len(population)))

    # Return the one with highest fitness
    return max(tournament, key=lambda g: g.fitness)


def genetic_token_selection(
    token_collection: TokenCollection,
    selection_size: int,
    recalculate_scores: Callable[[Set[str]], Dict[str, float]],
    population_size: int = 50,
    generations: int = 100,
    elite_count: int = 5,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> TokenCollection:
    """
    Genetic algorithm for selecting tokens:
    1. Create initial population
    2. Evaluate fitness
    3. Select, crossover, mutate to create new generation
    4. Repeat for specified generations

    Args:
        token_collection: Collection of tokens with scores and context
        selection_size: Number of tokens to select
        recalculate_scores: Function to recalculate scores based on selected tokens
        population_size: Size of the population
        generations: Number of generations to run
        elite_count: Number of top individuals to preserve unchanged
        mutation_rate: Probability of mutation for each token
        crossover_rate: Probability of crossover for each pair
        progress_callback: Optional callback for progress updates

    Returns:
        TokenCollection with assigned chords
    """
    # Make a working copy
    working_collection = copy.deepcopy(token_collection)

    # Initialize population
    population = [
        TokenSelectionGenome(working_collection, selection_size, random_init=True)
        for _ in range(population_size)
    ]

    # Add one "greedy" individual for a good starting point
    # Sort tokens by score and take the top selection_size
    sorted_tokens = sorted(
        working_collection.tokens, key=lambda t: t.score, reverse=True
    )
    greedy_tokens = set(t.original for t in sorted_tokens[:selection_size])
    population[0] = TokenSelectionGenome(
        working_collection, selection_size, greedy_tokens
    )

    # Calculate initial fitness
    for genome in population:
        genome.calculate_fitness(recalculate_scores)

    # Track best solution
    best_genome = max(population, key=lambda g: g.fitness)
    best_fitness = best_genome.fitness

    # Evolution loop
    for generation in range(generations):
        # Sort by fitness (descending)
        population.sort(key=lambda g: g.fitness, reverse=True)

        # Check if we have a new best
        if population[0].fitness > best_fitness:
            best_genome = population[0]
            best_fitness = best_genome.fitness

        # Report progress
        if progress_callback and generation % 5 == 0:
            progress_callback(
                generation,
                f"Generation {generation}/{generations}, Best fitness: {best_fitness:.2f}",
            )

        # Create next generation
        next_population = []

        # Elitism - carry over best individuals unchanged
        for i in range(min(elite_count, len(population))):
            next_population.append(population[i])

        # Fill the rest with crossover and mutation
        while len(next_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(population, 3)
            parent2 = tournament_selection(population, 3)

            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = parent1.crossover(parent2)

                # Mutation
                child1.mutate(mutation_rate)
                child2.mutate(mutation_rate)

                # Calculate fitness
                child1.calculate_fitness(recalculate_scores)
                child2.calculate_fitness(recalculate_scores)

                next_population.append(child1)
                if len(next_population) < population_size:
                    next_population.append(child2)
            else:
                # No crossover, just copy and mutate
                child = TokenSelectionGenome(
                    working_collection, selection_size, parent1.selected_tokens
                )
                child.mutate(mutation_rate)
                child.calculate_fitness(recalculate_scores)
                next_population.append(child)

        # Replace population
        population = next_population

    # Apply best solution to the collection
    best_genome = max(population, key=lambda g: g.fitness)

    # Create chords for selected tokens
    for token in working_collection.tokens:
        if token.original in best_genome.selected_tokens:
            token.assigned_chord = create_mock_chord(token.original)

    logger.info(
        f"Genetic selection completed with {len(best_genome.selected_tokens)} tokens"
    )
    logger.info(f"Best fitness: {best_genome.fitness:.2f}")

    return working_collection


# -----------------
# Main API
# -----------------


def select_tokens(
    token_collection: TokenCollection,
    max_tokens: int,
    recalculate_scores_func: Callable[[Set[str]], Dict[str, float]],
    algorithm: str = "greedy",
    algorithm_config: Optional[Dict] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> List[str]:
    """
    Select tokens for chord assignment using specified algorithm.

    Args:
        token_collection: Collection of tokens with scores
        max_tokens: Maximum number of tokens to select
        recalculate_scores_func: Function to recalculate scores
        algorithm: Algorithm to use ('greedy' or 'genetic')
        algorithm_config: Optional configuration parameters
        progress_callback: Optional callback for progress updates

    Returns:
        List of selected tokens
    """
    config = algorithm_config or {}

    if algorithm == "greedy":
        result_collection = greedy_token_selection(
            token_collection, max_tokens, recalculate_scores_func, progress_callback
        )
    elif algorithm == "genetic":
        # Extract genetic algorithm parameters
        population_size = config.get("population_size", 50)
        generations = config.get("generations", 100)
        elite_count = config.get("elite_count", 5)
        mutation_rate = config.get("mutation_rate", 0.1)
        crossover_rate = config.get("crossover_rate", 0.7)

        result_collection = genetic_token_selection(
            token_collection,
            max_tokens,
            recalculate_scores_func,
            population_size,
            generations,
            elite_count,
            mutation_rate,
            crossover_rate,
            progress_callback,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Extract selected tokens
    selected_tokens = []
    for token in result_collection.tokens:
        if hasattr(token, "assigned_chord") and token.assigned_chord is not None:
            selected_tokens.append(token.original)

    return selected_tokens


def save_selected_tokens(
    selected_tokens: List[str],
    token_collection: TokenCollection,
    output_path: str,
    algorithm_name: str,
) -> None:
    """
    Save selected tokens to a file.

    Args:
        selected_tokens: List of selected token strings
        token_collection: Original token collection
        output_path: Output file path
        algorithm_name: Name of algorithm used
    """
    # Create a new collection with only selected tokens
    token_map = {t.original: t for t in token_collection.tokens}
    selected_token_data = [token_map[t] for t in selected_tokens if t in token_map]

    # Create output collection
    output_collection = TokenCollection(
        name=f"{token_collection.name}_selected_{algorithm_name}",
        tokens=selected_token_data,
        ordered_by_frequency=False,
        source=token_collection.source,
    )

    # Save to file
    output_collection.save_to_file(output_path)
