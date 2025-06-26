"""
Implementation Agent - Forethought Phase Specialist

This agent specializes in supporting students during the forethought phase of 
Self-Regulated Learning, focusing on planning, strategy selection, and 
implementation guidance. It helps students think through problems systematically
before they begin coding.

Educational Focus Areas:
1. Algorithm Design Strategy: Helping students choose appropriate algorithms
2. Problem Decomposition: Breaking complex problems into manageable parts
3. Code Architecture Planning: Structuring solutions before implementation
4. Best Practices Guidance: Teaching industry-standard approaches
5. Strategic Thinking Development: Building systematic problem-solving skills

Pedagogical Approach:
- Socratic Method: Guiding students to discover solutions through questions
- Scaffolding: Providing structured support for complex planning tasks
- Metacognitive Development: Teaching students how to think about thinking
- Transfer Learning: Connecting new problems to known patterns and solutions
- Growth Mindset: Encouraging exploration and learning from mistakes

This agent generates responses that promote deep learning and strategic thinking
rather than providing direct solutions, fostering independent problem-solving skills.
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import re

from ..base_agent import BaseAgent, AgentType, AgentInput, AgentResponse, ResponseType
from ...utils.logging_utils import get_logger, LogContext, EventType
from ...config.settings import get_settings


class ImplementationStrategy(Enum):
    """Different implementation guidance strategies based on query type."""
    ALGORITHM_DESIGN = "algorithm_design"
    PROBLEM_DECOMPOSITION = "problem_decomposition"
    CODE_ARCHITECTURE = "code_architecture"
    PATTERN_RECOGNITION = "pattern_recognition"
    OPTIMIZATION_PLANNING = "optimization_planning"
    BEST_PRACTICES = "best_practices"
    STRATEGIC_THINKING = "strategic_thinking"


class ProgrammingDomain(Enum):
    """Programming domains for specialized guidance."""
    ALGORITHMS = "algorithms"
    DATA_STRUCTURES = "data_structures"
    OBJECT_ORIENTED = "object_oriented"
    WEB_DEVELOPMENT = "web_development"
    DATABASE = "database"
    SYSTEMS = "systems"
    MACHINE_LEARNING = "machine_learning"
    GENERAL = "general"


class ImplementationAgent(BaseAgent):
    """
    Specialized agent for forethought phase educational support.
    
    This agent helps students plan and strategize their coding approach,
    focusing on systematic thinking, problem decomposition, and strategic
    implementation planning. It promotes deep learning through guided
    discovery rather than direct solution provision.
    
    Key Capabilities:
    - Algorithm selection and design guidance
    - Problem decomposition and planning
    - Code architecture recommendations
    - Pattern recognition training
    - Best practices education
    - Strategic thinking development
    """
    
    def __init__(self, **kwargs):
        """Initialize the Implementation Agent with specialized capabilities."""
        super().__init__(**kwargs)
        
        # Domain detection patterns
        self.domain_patterns = self._initialize_domain_patterns()
        
        # Implementation strategy patterns
        self.strategy_patterns = self._initialize_strategy_patterns()
        
        # Educational concept mappings
        self.concept_mappings = self._initialize_concept_mappings()
        
        self.logger.log_event(
            EventType.COMPONENT_INIT,
            "Implementation Agent initialized",
            extra_data={"agent_type": "implementation", "specialization": "forethought_phase"}
        )
    
    def get_agent_type(self) -> AgentType:
        """Return the agent type for routing and identification."""
        return AgentType.IMPLEMENTATION
    
    def get_specialized_prompts(self) -> Dict[str, str]:
        """
        Return specialized prompts for implementation guidance.
        
        Returns:
            Dictionary of prompt templates for different scenarios
        """
        return {
            "system": """You are an expert programming implementation specialist and educational mentor. Your role is to guide students through the forethought phase of programming problem-solving, helping them plan, strategize, and think systematically about implementation approaches.

EDUCATIONAL APPROACH:
- Use the Socratic method to guide discovery rather than giving direct solutions
- Promote systematic thinking and problem decomposition
- Encourage exploration of multiple approaches and trade-offs
- Build metacognitive skills and strategic thinking abilities
- Connect new problems to familiar patterns and concepts

RESPONSE STYLE:
- Ask probing questions to guide thinking
- Provide structured frameworks for problem analysis
- Suggest step-by-step planning approaches
- Offer multiple perspectives and considerations
- Encourage reflection on choices and alternatives

Focus on building understanding and thinking skills, not just solving immediate problems.""",

            "algorithm_design": """Guide the student through systematic algorithm design:

1. PROBLEM UNDERSTANDING
- What exactly are we trying to achieve?
- What are the inputs, outputs, and constraints?
- Are there any edge cases to consider?

2. APPROACH EXPLORATION
- What similar problems have you solved before?
- What different strategies could work here?
- What are the trade-offs between approaches?

3. ALGORITHM SELECTION
- Which approach fits best given the constraints?
- How will this scale with larger inputs?
- What's the expected time and space complexity?

4. IMPLEMENTATION PLANNING
- How will you structure the solution?
- What helper functions or data structures are needed?
- What's your step-by-step implementation plan?""",

            "problem_decomposition": """Help the student break down complex problems:

1. IDENTIFY CORE COMPONENTS
- What are the main parts of this problem?
- Which components are independent vs. interdependent?
- What's the natural order for solving each part?

2. DEFINE INTERFACES
- How will different parts communicate?
- What data flows between components?
- What are the clear boundaries and responsibilities?

3. PRIORITIZE IMPLEMENTATION
- Which parts should be implemented first?
- What can be tested independently?
- How will you validate each component?

4. INTEGRATION STRATEGY
- How will you combine the parts?
- What integration challenges might arise?
- How will you test the complete solution?""",

            "best_practices": """Share relevant best practices and principles:

1. CODE ORGANIZATION
- How should you structure your code for clarity?
- What naming conventions will make it readable?
- How can you make it maintainable?

2. DESIGN PRINCIPLES
- What programming principles apply here?
- How can you make the code flexible and extensible?
- What patterns might be useful?

3. TESTING STRATEGY
- How will you verify correctness?
- What test cases should you consider?
- How can you test edge cases?

4. PERFORMANCE CONSIDERATIONS
- Where might performance bottlenecks occur?
- What optimizations might be worthwhile?
- How will you measure and improve performance?"""
        }
    
    def validate_specialized_input(self, agent_input: AgentInput) -> Tuple[bool, Optional[str]]:
        """
        Validate input for implementation agent requirements.
        
        Args:
            agent_input: Input to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Implementation agent can handle most planning queries
        if not agent_input.query.strip():
            return False, "Query cannot be empty"
        
        # Validate that this seems like a forethought phase query
        forethought_indicators = [
            "how to", "how do I", "how should I", "what's the best way",
            "approach", "strategy", "design", "implement", "plan",
            "structure", "organize", "architecture", "best practice"
        ]
        
        query_lower = agent_input.query.lower()
        has_forethought_indicators = any(indicator in query_lower for indicator in forethought_indicators)
        
        # Check for debugging indicators (should go to debugging agent)
        debugging_indicators = [
            "error", "bug", "fix", "debug", "wrong", "doesn't work",
            "not working", "broken", "crash", "exception", "fails"
        ]
        
        has_debugging_indicators = any(indicator in query_lower for indicator in debugging_indicators)
        
        # If it has debugging indicators but no forethought indicators, might be misrouted
        if has_debugging_indicators and not has_forethought_indicators and not agent_input.code_snippet:
            return True, None  # Still accept but log warning
        
        return True, None
    
    def process_specialized_response(self, 
                                   agent_input: AgentInput,
                                   rag_context: Optional[List[str]],
                                   base_response: str) -> AgentResponse:
        """
        Process and enhance the response with implementation-specific guidance.
        
        Args:
            agent_input: Original student input
            rag_context: Retrieved educational context
            base_response: Base response from LLM
            
        Returns:
            Enhanced AgentResponse with educational metadata
        """
        # Detect domain and strategy
        domain = self._detect_programming_domain(agent_input.query)
        strategy = self._determine_implementation_strategy(agent_input)
        
        # Assess response quality and educational value
        response_quality = self._assess_response_quality(base_response)
        educational_value = self._assess_educational_value(base_response, agent_input)
        
        # Generate educational metadata
        concepts_covered = self._identify_concepts_covered(agent_input.query, base_response)
        learning_objectives = self._generate_learning_objectives(strategy, concepts_covered)
        
        # Create suggested follow-up questions
        follow_up_questions = self._generate_follow_up_questions(agent_input, strategy, domain)
        
        # Build comprehensive educational metadata
        educational_metadata = self.create_educational_metadata(
            agent_input=agent_input,
            confidence=response_quality,
            concepts_covered=concepts_covered,
            learning_objectives=learning_objectives
        )
        
        # Add implementation-specific metadata
        educational_metadata.update({
            "implementation_strategy": strategy.value,
            "programming_domain": domain.value,
            "educational_value_score": educational_value,
            "promotes_strategic_thinking": self._promotes_strategic_thinking(base_response),
            "encourages_decomposition": self._encourages_decomposition(base_response),
            "socratic_elements": self._has_socratic_elements(base_response),
            "complexity_appropriate": self._is_complexity_appropriate(base_response, agent_input.student_level)
        })
        
        # Determine response type based on strategy
        response_type = self._determine_response_type(strategy)
        
        return AgentResponse(
            content=base_response,
            response_type=response_type,
            agent_type=self.get_agent_type(),
            confidence=response_quality,
            educational_metadata=educational_metadata,
            rag_context=rag_context,
            suggested_follow_up=follow_up_questions
        )
    
    def _initialize_domain_patterns(self) -> Dict[ProgrammingDomain, List[str]]:
        """Initialize patterns for programming domain detection."""
        return {
            ProgrammingDomain.ALGORITHMS: [
                "algorithm", "sorting", "searching", "traversal", "complexity",
                "big o", "optimization", "efficient", "binary search", "merge sort",
                "quicksort", "dynamic programming", "greedy", "divide and conquer"
            ],
            ProgrammingDomain.DATA_STRUCTURES: [
                "array", "list", "stack", "queue", "tree", "graph", "hash",
                "linked list", "binary tree", "heap", "priority queue",
                "dictionary", "set", "data structure", "node", "pointer"
            ],
            ProgrammingDomain.OBJECT_ORIENTED: [
                "class", "object", "inheritance", "polymorphism", "encapsulation",
                "abstraction", "interface", "constructor", "method", "attribute",
                "design pattern", "singleton", "factory", "observer"
            ],
            ProgrammingDomain.WEB_DEVELOPMENT: [
                "web", "html", "css", "javascript", "api", "rest", "http",
                "server", "client", "frontend", "backend", "database", "json",
                "ajax", "react", "node", "express", "django", "flask"
            ],
            ProgrammingDomain.DATABASE: [
                "database", "sql", "query", "table", "join", "index",
                "mysql", "postgresql", "mongodb", "nosql", "schema",
                "relational", "transaction", "acid", "crud"
            ],
            ProgrammingDomain.MACHINE_LEARNING: [
                "machine learning", "ml", "ai", "neural network", "model",
                "training", "prediction", "classification", "regression",
                "clustering", "deep learning", "tensorflow", "pytorch"
            ]
        }
    
    def _initialize_strategy_patterns(self) -> Dict[ImplementationStrategy, List[str]]:
        """Initialize patterns for implementation strategy detection."""
        return {
            ImplementationStrategy.ALGORITHM_DESIGN: [
                "algorithm", "approach", "method", "technique", "solve",
                "efficient", "optimal", "complexity", "performance"
            ],
            ImplementationStrategy.PROBLEM_DECOMPOSITION: [
                "break down", "decompose", "structure", "organize", "divide",
                "modular", "components", "parts", "steps"
            ],
            ImplementationStrategy.CODE_ARCHITECTURE: [
                "architecture", "design", "structure", "organize", "layout",
                "framework", "pattern", "hierarchy", "organization"
            ],
            ImplementationStrategy.PATTERN_RECOGNITION: [
                "pattern", "similar", "like", "previous", "seen before",
                "familiar", "template", "example", "model"
            ],
            ImplementationStrategy.OPTIMIZATION_PLANNING: [
                "optimize", "improve", "faster", "efficient", "performance",
                "memory", "speed", "scalable", "bottleneck"
            ],
            ImplementationStrategy.BEST_PRACTICES: [
                "best practice", "standard", "convention", "proper way",
                "recommended", "guideline", "principle", "clean code"
            ]
        }
    
    def _initialize_concept_mappings(self) -> Dict[str, List[str]]:
        """Initialize mappings between keywords and programming concepts."""
        return {
            "searching": ["linear search", "binary search", "hash lookup", "indexing"],
            "sorting": ["bubble sort", "merge sort", "quicksort", "heap sort", "comparison"],
            "recursion": ["base case", "recursive case", "call stack", "divide and conquer"],
            "iteration": ["loops", "for loop", "while loop", "iterators", "traversal"],
            "data_flow": ["input", "processing", "output", "transformation", "pipeline"],
            "error_handling": ["validation", "edge cases", "error checking", "robustness"],
            "modularity": ["functions", "classes", "modules", "separation of concerns"],
            "abstraction": ["interfaces", "encapsulation", "information hiding", "simplification"]
        }
    
    def _detect_programming_domain(self, query: str) -> ProgrammingDomain:
        """
        Detect the programming domain of the query.
        
        Args:
            query: Student query
            
        Returns:
            Detected programming domain
        """
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score, default to GENERAL
        if domain_scores and max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        else:
            return ProgrammingDomain.GENERAL
    
    def _determine_implementation_strategy(self, agent_input: AgentInput) -> ImplementationStrategy:
        """
        Determine the best implementation strategy for the query.
        
        Args:
            agent_input: Student input
            
        Returns:
            Recommended implementation strategy
        """
        query_lower = agent_input.query.lower()
        strategy_scores = {}
        
        for strategy, keywords in self.strategy_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            strategy_scores[strategy] = score
        
        # Special logic for strategy selection
        if "break" in query_lower and "down" in query_lower:
            return ImplementationStrategy.PROBLEM_DECOMPOSITION
        
        if "best" in query_lower and ("practice" in query_lower or "way" in query_lower):
            return ImplementationStrategy.BEST_PRACTICES
        
        if "design" in query_lower or "architecture" in query_lower:
            return ImplementationStrategy.CODE_ARCHITECTURE
        
        if "algorithm" in query_lower or "approach" in query_lower:
            return ImplementationStrategy.ALGORITHM_DESIGN
        
        # Return strategy with highest score, default to STRATEGIC_THINKING
        if strategy_scores and max(strategy_scores.values()) > 0:
            return max(strategy_scores, key=strategy_scores.get)
        else:
            return ImplementationStrategy.STRATEGIC_THINKING
    
    def _assess_response_quality(self, response: str) -> float:
        """
        Assess the quality of the generated response.
        
        Args:
            response: Generated response
            
        Returns:
            Quality score from 0.0 to 1.0
        """
        quality_score = 0.5  # Base score
        
        # Check for educational elements
        educational_indicators = [
            "consider", "think about", "what if", "how might",
            "approach", "strategy", "plan", "design", "structure"
        ]
        
        socratic_indicators = [
            "?", "what", "how", "why", "which", "when", "where"
        ]
        
        structure_indicators = [
            "first", "next", "then", "finally", "step", "phase"
        ]
        
        response_lower = response.lower()
        
        # Bonus for educational language
        quality_score += min(0.2, sum(0.05 for ind in educational_indicators if ind in response_lower))
        
        # Bonus for Socratic questioning
        question_count = response.count("?")
        quality_score += min(0.15, question_count * 0.03)
        
        # Bonus for structured approach
        quality_score += min(0.15, sum(0.03 for ind in structure_indicators if ind in response_lower))
        
        # Check response length (should be substantive)
        if len(response) > 200:
            quality_score += 0.1
        if len(response) > 500:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _assess_educational_value(self, response: str, agent_input: AgentInput) -> float:
        """
        Assess the educational value of the response.
        
        Args:
            response: Generated response
            agent_input: Original input
            
        Returns:
            Educational value score from 0.0 to 1.0
        """
        value_score = 0.0
        response_lower = response.lower()
        
        # Check for learning-promoting elements
        learning_elements = {
            "metacognitive": ["think about thinking", "consider your approach", "reflect on"],
            "transfer": ["similar to", "like when", "pattern", "previous experience"],
            "exploration": ["try", "experiment", "explore", "consider alternatives"],
            "understanding": ["understand", "concept", "principle", "why", "because"]
        }
        
        for element_type, indicators in learning_elements.items():
            for indicator in indicators:
                if indicator in response_lower:
                    value_score += 0.15
        
        # Check for guidance rather than solutions
        guidance_indicators = [
            "consider", "think", "plan", "design", "approach",
            "strategy", "steps", "process", "method"
        ]
        
        solution_indicators = [
            "here's the answer", "the solution is", "copy this",
            "just use", "simply do"
        ]
        
        guidance_count = sum(1 for ind in guidance_indicators if ind in response_lower)
        solution_count = sum(1 for ind in solution_indicators if ind in response_lower)
        
        value_score += min(0.3, guidance_count * 0.05)
        value_score -= min(0.2, solution_count * 0.1)  # Penalize direct solutions
        
        return max(0.0, min(1.0, value_score))
    
    def _identify_concepts_covered(self, query: str, response: str) -> List[str]:
        """
        Identify programming concepts covered in the interaction.
        
        Args:
            query: Student query
            response: Generated response
            
        Returns:
            List of programming concepts covered
        """
        concepts = []
        combined_text = (query + " " + response).lower()
        
        # Check for concepts in our mappings
        for concept_category, related_concepts in self.concept_mappings.items():
            if any(concept in combined_text for concept in related_concepts):
                concepts.append(concept_category)
        
        # Check for specific algorithm concepts
        algorithm_concepts = [
            "binary search", "linear search", "sorting", "recursion",
            "iteration", "dynamic programming", "greedy algorithm",
            "divide and conquer", "breadth first", "depth first"
        ]
        
        for concept in algorithm_concepts:
            if concept in combined_text:
                concepts.append(concept.replace(" ", "_"))
        
        return list(set(concepts))  # Remove duplicates
    
    def _generate_learning_objectives(self, 
                                    strategy: ImplementationStrategy,
                                    concepts: List[str]) -> List[str]:
        """
        Generate learning objectives based on strategy and concepts.
        
        Args:
            strategy: Implementation strategy used
            concepts: Concepts covered
            
        Returns:
            List of learning objectives
        """
        objectives = []
        
        # Strategy-based objectives
        strategy_objectives = {
            ImplementationStrategy.ALGORITHM_DESIGN: [
                "Analyze problem requirements systematically",
                "Compare different algorithmic approaches",
                "Consider time and space complexity trade-offs"
            ],
            ImplementationStrategy.PROBLEM_DECOMPOSITION: [
                "Break complex problems into manageable components",
                "Identify clear interfaces between components",
                "Plan implementation order and dependencies"
            ],
            ImplementationStrategy.CODE_ARCHITECTURE: [
                "Design clean and maintainable code structure",
                "Apply appropriate design patterns",
                "Consider future extensibility and modification"
            ],
            ImplementationStrategy.BEST_PRACTICES: [
                "Apply industry-standard coding conventions",
                "Write readable and documented code",
                "Consider testing and debugging strategies"
            ]
        }
        
        objectives.extend(strategy_objectives.get(strategy, []))
        
        # Concept-based objectives
        if "recursion" in concepts:
            objectives.append("Understand recursive problem-solving patterns")
        if "data_structures" in concepts:
            objectives.append("Select appropriate data structures for problems")
        if "algorithms" in concepts:
            objectives.append("Analyze and design efficient algorithms")
        
        return objectives[:5]  # Limit to top 5 objectives
    
    def _generate_follow_up_questions(self, 
                                    agent_input: AgentInput,
                                    strategy: ImplementationStrategy,
                                    domain: ProgrammingDomain) -> List[str]:
        """
        Generate relevant follow-up questions to promote deeper learning.
        
        Args:
            agent_input: Original input
            strategy: Implementation strategy used
            domain: Programming domain
            
        Returns:
            List of follow-up questions
        """
        questions = []
        
        # Strategy-specific questions
        if strategy == ImplementationStrategy.ALGORITHM_DESIGN:
            questions.extend([
                "What would happen if the input size was much larger?",
                "Are there any edge cases you should consider?",
                "How would you test this algorithm thoroughly?"
            ])
        elif strategy == ImplementationStrategy.PROBLEM_DECOMPOSITION:
            questions.extend([
                "Which component would you implement first and why?",
                "How will you handle communication between components?",
                "What happens if one component fails?"
            ])
        elif strategy == ImplementationStrategy.CODE_ARCHITECTURE:
            questions.extend([
                "How would you modify this design for different requirements?",
                "What design patterns might be applicable here?",
                "How will you ensure the code remains maintainable?"
            ])
        
        # Domain-specific questions
        if domain == ProgrammingDomain.ALGORITHMS:
            questions.append("Can you think of a more efficient approach?")
        elif domain == ProgrammingDomain.DATA_STRUCTURES:
            questions.append("Would a different data structure work better?")
        elif domain == ProgrammingDomain.OBJECT_ORIENTED:
            questions.append("How would inheritance or composition help here?")
        
        # General learning questions
        questions.extend([
            "What similar problems have you solved before?",
            "How will you verify that your solution works correctly?",
            "What would you do differently if you started over?"
        ])
        
        return questions[:4]  # Limit to top 4 questions
    
    def _promotes_strategic_thinking(self, response: str) -> bool:
        """Check if response promotes strategic thinking."""
        strategic_indicators = [
            "consider", "think", "plan", "strategy", "approach",
            "alternatives", "trade-offs", "options", "choices"
        ]
        response_lower = response.lower()
        return sum(1 for ind in strategic_indicators if ind in response_lower) >= 2
    
    def _encourages_decomposition(self, response: str) -> bool:
        """Check if response encourages problem decomposition."""
        decomposition_indicators = [
            "break", "divide", "component", "part", "step",
            "module", "separate", "organize", "structure"
        ]
        response_lower = response.lower()
        return any(ind in response_lower for ind in decomposition_indicators)
    
    def _has_socratic_elements(self, response: str) -> bool:
        """Check if response uses Socratic questioning method."""
        question_count = response.count("?")
        socratic_phrases = [
            "what if", "how might", "why do you think",
            "what would happen", "consider", "think about"
        ]
        response_lower = response.lower()
        
        has_questions = question_count >= 2
        has_socratic_language = any(phrase in response_lower for phrase in socratic_phrases)
        
        return has_questions or has_socratic_language
    
    def _is_complexity_appropriate(self, response: str, student_level: Optional[str]) -> bool:
        """Check if response complexity matches student level."""
        if not student_level:
            return True  # Cannot assess without level
        
        response_lower = response.lower()
        
        advanced_terms = [
            "complexity", "optimization", "algorithm", "data structure",
            "pattern", "architecture", "scalability", "performance"
        ]
        
        beginner_terms = [
            "step", "simple", "basic", "start", "first", "easy", "begin"
        ]
        
        advanced_count = sum(1 for term in advanced_terms if term in response_lower)
        beginner_count = sum(1 for term in beginner_terms if term in response_lower)
        
        if student_level == "beginner":
            return beginner_count >= advanced_count
        elif student_level == "advanced":
            return advanced_count > 0
        else:  # intermediate
            return True  # Generally appropriate
    
    def _determine_response_type(self, strategy: ImplementationStrategy) -> ResponseType:
        """Determine appropriate response type based on strategy."""
        strategy_response_mapping = {
            ImplementationStrategy.ALGORITHM_DESIGN: ResponseType.GUIDANCE,
            ImplementationStrategy.PROBLEM_DECOMPOSITION: ResponseType.GUIDANCE,
            ImplementationStrategy.CODE_ARCHITECTURE: ResponseType.GUIDANCE,
            ImplementationStrategy.PATTERN_RECOGNITION: ResponseType.EXPLANATION,
            ImplementationStrategy.OPTIMIZATION_PLANNING: ResponseType.GUIDANCE,
            ImplementationStrategy.BEST_PRACTICES: ResponseType.EXPLANATION,
            ImplementationStrategy.STRATEGIC_THINKING: ResponseType.GUIDANCE
        }
        
        return strategy_response_mapping.get(strategy, ResponseType.GUIDANCE)


if __name__ == "__main__":
    # Implementation agent test
    try:
        from ...utils.api_utils import get_openai_client
        
        agent = ImplementationAgent()
        
        # Test cases for different implementation scenarios
        test_cases = [
            AgentInput(
                query="How do I implement a binary search algorithm efficiently?",
                student_level="intermediate"
            ),
            AgentInput(
                query="What's the best approach to design a class hierarchy for a game?",
                student_level="advanced"
            ),
            AgentInput(
                query="How should I structure my web application backend?",
                student_level="intermediate"
            )
        ]
        
        for i, test_input in enumerate(test_cases):
            print(f"\nTest Case {i+1}: {test_input.query}")
            
            response = agent.process_query(test_input)
            
            print(f"Response Type: {response.response_type}")
            print(f"Confidence: {response.confidence:.3f}")
            print(f"Strategy: {response.educational_metadata.get('implementation_strategy')}")
            print(f"Domain: {response.educational_metadata.get('programming_domain')}")
            print(f"Educational Value: {response.educational_metadata.get('educational_value_score', 0):.3f}")
            print(f"Concepts: {response.educational_metadata.get('concepts_covered', [])}")
            print(f"Follow-up Questions: {len(response.suggested_follow_up or [])}")
        
        # Performance stats
        stats = agent.get_performance_stats()
        print(f"\nAgent Performance: {stats}")
        
        print("✅ Implementation agent test completed successfully!")
        
    except Exception as e:
        print(f"❌ Implementation agent test failed: {e}")
        import traceback
        traceback.print_exc()
