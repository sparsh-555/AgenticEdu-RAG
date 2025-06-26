"""
Debugging Agent - Performance Phase Specialist

This agent specializes in supporting students during the performance phase of 
Self-Regulated Learning, focusing on error diagnosis, debugging strategies, 
and systematic problem-solving for existing code issues.

Educational Focus Areas:
1. Error Analysis: Teaching students to read and understand error messages
2. Systematic Debugging: Building methodical troubleshooting skills
3. Code Inspection: Developing skills to identify bugs through code review
4. Testing Strategies: Teaching effective testing and validation approaches
5. Self-Monitoring: Building metacognitive debugging awareness

Pedagogical Approach:
- Guided Discovery: Leading students to find bugs themselves rather than fixing directly
- Systematic Method: Teaching step-by-step debugging processes
- Error Pattern Recognition: Helping students recognize common error types
- Metacognitive Development: Building awareness of debugging strategies
- Resilience Building: Encouraging persistence and learning from mistakes

This agent promotes deep debugging skills and systematic thinking about 
code behavior, fostering independent problem-solving capabilities.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import re
import ast

from ..base_agent import BaseAgent, AgentType, AgentInput, AgentResponse, ResponseType
from ...utils.logging_utils import get_logger, LogContext, EventType
from ...config.settings import get_settings


class DebuggingStrategy(Enum):
    """Different debugging strategies based on error type and context."""
    ERROR_MESSAGE_ANALYSIS = "error_message_analysis"
    CODE_INSPECTION = "code_inspection"
    LOGICAL_REASONING = "logical_reasoning"
    SYSTEMATIC_TESTING = "systematic_testing"
    TRACE_EXECUTION = "trace_execution"
    PATTERN_RECOGNITION = "pattern_recognition"
    DIVIDE_AND_CONQUER = "divide_and_conquer"


class ErrorCategory(Enum):
    """Categories of programming errors for specialized handling."""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    TYPE_ERROR = "type_error"
    INDEX_ERROR = "index_error"
    NAME_ERROR = "name_error"
    ATTRIBUTE_ERROR = "attribute_error"
    IMPORT_ERROR = "import_error"
    PERFORMANCE_ISSUE = "performance_issue"
    DESIGN_FLAW = "design_flaw"
    UNKNOWN = "unknown"


class DebuggingComplexity(Enum):
    """Complexity levels for debugging guidance adaptation."""
    SIMPLE = "simple"          # Single obvious error
    MODERATE = "moderate"      # Multiple errors or unclear cause
    COMPLEX = "complex"        # Systemic issues or advanced debugging needed


class DebuggingAgent(BaseAgent):
    """
    Specialized agent for performance phase educational support.
    
    This agent helps students debug existing code, understand errors,
    and develop systematic approaches to problem-solving. It focuses
    on building debugging skills and error analysis capabilities rather
    than providing direct fixes.
    
    Key Capabilities:
    - Error message interpretation and analysis
    - Systematic debugging strategy guidance
    - Code inspection and logical error detection
    - Testing and validation approaches
    - Performance issue identification
    - Debugging methodology education
    """
    
    def __init__(self, **kwargs):
        """Initialize the Debugging Agent with specialized capabilities."""
        super().__init__(**kwargs)
        
        # Error pattern recognition
        self.error_patterns = self._initialize_error_patterns()
        
        # Common debugging strategies
        self.debugging_strategies = self._initialize_debugging_strategies()
        
        # Code analysis patterns
        self.code_analysis_patterns = self._initialize_code_patterns()
        
        # Common error fixes (for educational guidance, not direct solutions)
        self.error_fix_guidance = self._initialize_fix_guidance()
        
        self.logger.log_event(
            EventType.COMPONENT_INIT,
            "Debugging Agent initialized",
            extra_data={"agent_type": "debugging", "specialization": "performance_phase"}
        )
    
    def get_agent_type(self) -> AgentType:
        """Return the agent type for routing and identification."""
        return AgentType.DEBUGGING
    
    def get_specialized_prompts(self) -> Dict[str, str]:
        """
        Return specialized prompts for debugging guidance.
        
        Returns:
            Dictionary of prompt templates for different debugging scenarios
        """
        return {
            "system": """You are an expert programming debugging specialist and educational mentor. Your role is to guide students through systematic debugging and error-solving processes, helping them develop independent problem-solving skills.

EDUCATIONAL APPROACH:
- Teach systematic debugging methodology rather than providing direct fixes
- Help students understand error messages and their underlying causes
- Guide students to discover bugs through structured inquiry
- Build metacognitive awareness of debugging strategies
- Encourage systematic testing and validation approaches

DEBUGGING METHODOLOGY:
1. UNDERSTAND THE ERROR: What exactly is happening vs. what should happen?
2. ANALYZE THE MESSAGE: What is the error telling us specifically?
3. LOCATE THE PROBLEM: Where in the code might the issue originate?
4. HYPOTHESIZE CAUSES: What could be causing this behavior?
5. TEST SYSTEMATICALLY: How can we verify our hypotheses?
6. APPLY FIXES: What changes address the root cause?

Focus on building debugging skills and understanding, not just solving immediate problems.""",

            "error_analysis": """Guide the student through systematic error analysis:

1. ERROR MESSAGE INTERPRETATION
- What specific error type is this?
- What line or component does it point to?
- What is the error message telling us in plain language?

2. CONTEXT ANALYSIS
- When does this error occur? (always, sometimes, specific inputs?)
- What was the code trying to do when it failed?
- Are there patterns to when it works vs. fails?

3. ROOT CAUSE INVESTIGATION
- What might be causing this specific error?
- Are there related issues that might contribute?
- What assumptions in the code might be incorrect?

4. DEBUGGING STRATEGY
- How can we test our hypotheses about the cause?
- What debugging tools or techniques would help?
- What's the most systematic way to isolate the problem?""",

            "code_inspection": """Help the student systematically inspect their code:

1. CODE STRUCTURE REVIEW
- Does the overall logic flow make sense?
- Are there any obvious structural issues?
- Do variable names and functions match their purposes?

2. LINE-BY-LINE ANALYSIS
- What does each line actually do vs. what it should do?
- Are there any suspicious patterns or anomalies?
- Where might the code behavior diverge from expectations?

3. DATA FLOW TRACKING
- How does data move through the program?
- Where might data get corrupted or transformed unexpectedly?
- Are all variables properly initialized and updated?

4. EDGE CASE CONSIDERATION
- What happens with empty inputs, null values, or boundary conditions?
- Are there assumptions that might not hold in all cases?
- What could go wrong with different types of input?""",

            "systematic_testing": """Guide the student in systematic testing approaches:

1. TEST CASE DESIGN
- What are the simplest test cases to start with?
- How can we isolate different components or functions?
- What edge cases should we specifically test?

2. DEBUGGING OUTPUT
- What information would help us understand what's happening?
- Where should we add print statements or logging?
- How can we trace the program's execution?

3. INCREMENTAL TESTING
- How can we test parts of the code independently?
- What's the minimal example that reproduces the problem?
- How can we gradually build up to the full complexity?

4. VALIDATION STRATEGY
- How will we know when the bug is truly fixed?
- What tests should pass consistently?
- How can we prevent similar issues in the future?""",

            "performance_debugging": """Help debug performance and efficiency issues:

1. PERFORMANCE MEASUREMENT
- How can we measure the actual performance?
- Where are the potential bottlenecks in the code?
- What operations might be unexpectedly slow?

2. ALGORITHMIC ANALYSIS
- Is the algorithm fundamentally efficient for the problem size?
- Are there redundant or unnecessary operations?
- Could a different approach be more efficient?

3. RESOURCE USAGE
- Is memory being used efficiently?
- Are there memory leaks or excessive allocations?
- How does performance scale with input size?

4. OPTIMIZATION STRATEGY
- What optimizations would have the biggest impact?
- How can we validate that optimizations actually help?
- What trade-offs should we consider?"""
        }
    
    def validate_specialized_input(self, agent_input: AgentInput) -> Tuple[bool, Optional[str]]:
        """
        Validate input for debugging agent requirements.
        
        Args:
            agent_input: Input to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not agent_input.query.strip():
            return False, "Query cannot be empty"
        
        # Check for debugging indicators
        debugging_indicators = [
            "error", "bug", "fix", "debug", "wrong", "doesn't work",
            "not working", "broken", "crash", "exception", "fails",
            "problem", "issue", "trouble"
        ]
        
        query_lower = agent_input.query.lower()
        has_debugging_indicators = any(indicator in query_lower for indicator in debugging_indicators)
        
        # Strong indicators: error message or code snippet with problems
        has_strong_indicators = (
            agent_input.error_message is not None or
            (agent_input.code_snippet is not None and has_debugging_indicators)
        )
        
        # This agent can handle most debugging queries, but log if questionable
        if not has_debugging_indicators and not has_strong_indicators:
            self.logger.log_event(
                EventType.WARNING_ISSUED,
                "Query may not be appropriate for debugging agent",
                extra_data={"query_preview": agent_input.query[:100]}
            )
        
        return True, None
    
    def process_specialized_response(self, 
                                   agent_input: AgentInput,
                                   rag_context: Optional[List[str]],
                                   base_response: str) -> AgentResponse:
        """
        Process and enhance the response with debugging-specific guidance.
        
        Args:
            agent_input: Original student input
            rag_context: Retrieved educational context
            base_response: Base response from LLM
            
        Returns:
            Enhanced AgentResponse with debugging metadata
        """
        # Analyze the error and code context
        error_category = self._categorize_error(agent_input)
        debugging_strategy = self._determine_debugging_strategy(agent_input, error_category)
        complexity = self._assess_debugging_complexity(agent_input)
        
        # Analyze code if provided
        code_analysis = self._analyze_code_snippet(agent_input.code_snippet) if agent_input.code_snippet else {}
        
        # Assess response quality for debugging context
        response_quality = self._assess_debugging_response_quality(base_response, agent_input)
        educational_value = self._assess_debugging_educational_value(base_response)
        
        # Generate debugging-specific educational metadata
        concepts_covered = self._identify_debugging_concepts(agent_input, base_response)
        learning_objectives = self._generate_debugging_learning_objectives(debugging_strategy, error_category)
        
        # Create debugging-specific follow-up questions
        follow_up_questions = self._generate_debugging_follow_up(agent_input, debugging_strategy, error_category)
        
        # Build comprehensive educational metadata
        educational_metadata = self.create_educational_metadata(
            agent_input=agent_input,
            confidence=response_quality,
            concepts_covered=concepts_covered,
            learning_objectives=learning_objectives
        )
        
        # Add debugging-specific metadata
        educational_metadata.update({
            "debugging_strategy": debugging_strategy.value,
            "error_category": error_category.value,
            "debugging_complexity": complexity.value,
            "systematic_approach": self._uses_systematic_approach(base_response),
            "teaches_debugging_skills": self._teaches_debugging_skills(base_response),
            "encourages_testing": self._encourages_testing(base_response),
            "builds_error_understanding": self._builds_error_understanding(base_response),
            **code_analysis
        })
        
        # Determine response type
        response_type = self._determine_debugging_response_type(debugging_strategy, error_category)
        
        return AgentResponse(
            content=base_response,
            response_type=response_type,
            agent_type=self.get_agent_type(),
            confidence=response_quality,
            educational_metadata=educational_metadata,
            rag_context=rag_context,
            suggested_follow_up=follow_up_questions
        )
    
    def _initialize_error_patterns(self) -> Dict[ErrorCategory, List[str]]:
        """Initialize patterns for error category detection."""
        return {
            ErrorCategory.SYNTAX_ERROR: [
                "syntaxerror", "invalid syntax", "unexpected token", "missing colon",
                "indentation", "unmatched parentheses", "missing quote"
            ],
            ErrorCategory.RUNTIME_ERROR: [
                "runtimeerror", "recursionerror", "maximum recursion depth",
                "stack overflow", "memory error", "timeout"
            ],
            ErrorCategory.TYPE_ERROR: [
                "typeerror", "unsupported operand", "not subscriptable",
                "object is not callable", "argument of type"
            ],
            ErrorCategory.INDEX_ERROR: [
                "indexerror", "list index out of range", "string index out of range",
                "index out of bounds"
            ],
            ErrorCategory.NAME_ERROR: [
                "nameerror", "not defined", "undefined variable",
                "unknown identifier"
            ],
            ErrorCategory.ATTRIBUTE_ERROR: [
                "attributeerror", "has no attribute", "object has no attribute",
                "module has no attribute"
            ],
            ErrorCategory.IMPORT_ERROR: [
                "importerror", "modulenotfounderror", "no module named",
                "cannot import", "import error"
            ],
            ErrorCategory.LOGIC_ERROR: [
                "wrong output", "incorrect result", "not working as expected",
                "logic error", "wrong answer", "unexpected behavior"
            ],
            ErrorCategory.PERFORMANCE_ISSUE: [
                "slow", "performance", "timeout", "taking too long",
                "inefficient", "memory usage", "optimization"
            ]
        }
    
    def _initialize_debugging_strategies(self) -> Dict[str, List[str]]:
        """Initialize debugging strategy recommendations."""
        return {
            "systematic_approach": [
                "Start with the error message - what exactly is it telling us?",
                "Identify the line number and specific location of the error",
                "Understand what the code was trying to do at that point",
                "Form hypotheses about what might be causing the issue",
                "Test each hypothesis systematically"
            ],
            "code_inspection": [
                "Read through the code line by line",
                "Check variable names and their usage",
                "Verify data types and conversions",
                "Look for boundary conditions and edge cases",
                "Trace the execution flow mentally"
            ],
            "testing_strategy": [
                "Create minimal test cases that reproduce the error",
                "Test with simple, known inputs first",
                "Add debugging output to trace execution",
                "Isolate different parts of the code",
                "Verify assumptions with print statements"
            ]
        }
    
    def _initialize_code_patterns(self) -> Dict[str, List[str]]:
        """Initialize common problematic code patterns."""
        return {
            "index_issues": [
                "range(len(", "for i in range(len(", "[i+1]", "[i-1]",
                "while i <", "while i <="
            ],
            "loop_issues": [
                "while True:", "while", "for", "infinite loop",
                "loop condition", "increment", "decrement"
            ],
            "variable_issues": [
                "undefined", "not initialized", "scope", "global",
                "local variable", "assignment"
            ],
            "function_issues": [
                "return", "parameters", "arguments", "function call",
                "missing return", "wrong arguments"
            ]
        }
    
    def _initialize_fix_guidance(self) -> Dict[ErrorCategory, List[str]]:
        """Initialize educational guidance for different error types."""
        return {
            ErrorCategory.INDEX_ERROR: [
                "Check if the index is within the valid range (0 to len-1)",
                "Consider if you're accessing elements that might not exist",
                "Use len() to verify the size of your data structure",
                "Check loop boundaries and off-by-one errors"
            ],
            ErrorCategory.TYPE_ERROR: [
                "Verify the data types of variables involved in the operation",
                "Check if you need to convert between types (int, str, float)",
                "Ensure you're using the right methods for the data type",
                "Look for mixing incompatible operations"
            ],
            ErrorCategory.NAME_ERROR: [
                "Check if the variable name is spelled correctly",
                "Verify the variable is defined before use",
                "Consider variable scope (local vs global)",
                "Make sure imports are correct if using external modules"
            ],
            ErrorCategory.LOGIC_ERROR: [
                "Trace through your algorithm step by step",
                "Check if your conditions and loops work as expected",
                "Verify that your logic matches the problem requirements",
                "Test with simple examples where you know the expected output"
            ]
        }
    
    def _categorize_error(self, agent_input: AgentInput) -> ErrorCategory:
        """
        Categorize the error based on error message and context.
        
        Args:
            agent_input: Student input with potential error information
            
        Returns:
            Categorized error type
        """
        # Check error message first
        if agent_input.error_message:
            error_msg_lower = agent_input.error_message.lower()
            
            for category, patterns in self.error_patterns.items():
                if any(pattern in error_msg_lower for pattern in patterns):
                    return category
        
        # Check query content for error indicators
        query_lower = agent_input.query.lower()
        
        for category, patterns in self.error_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return category
        
        # Default categorization based on context
        if agent_input.code_snippet and not agent_input.error_message:
            return ErrorCategory.LOGIC_ERROR
        
        return ErrorCategory.UNKNOWN
    
    def _determine_debugging_strategy(self, 
                                    agent_input: AgentInput,
                                    error_category: ErrorCategory) -> DebuggingStrategy:
        """
        Determine the most appropriate debugging strategy.
        
        Args:
            agent_input: Student input
            error_category: Categorized error type
            
        Returns:
            Recommended debugging strategy
        """
        # Strategy based on error category
        if error_category in [ErrorCategory.SYNTAX_ERROR, ErrorCategory.NAME_ERROR, ErrorCategory.IMPORT_ERROR]:
            return DebuggingStrategy.ERROR_MESSAGE_ANALYSIS
        
        if error_category in [ErrorCategory.INDEX_ERROR, ErrorCategory.TYPE_ERROR, ErrorCategory.ATTRIBUTE_ERROR]:
            return DebuggingStrategy.CODE_INSPECTION
        
        if error_category == ErrorCategory.LOGIC_ERROR:
            return DebuggingStrategy.LOGICAL_REASONING
        
        if error_category == ErrorCategory.PERFORMANCE_ISSUE:
            return DebuggingStrategy.SYSTEMATIC_TESTING
        
        # Strategy based on available information
        if agent_input.error_message and not agent_input.code_snippet:
            return DebuggingStrategy.ERROR_MESSAGE_ANALYSIS
        
        if agent_input.code_snippet and agent_input.error_message:
            return DebuggingStrategy.TRACE_EXECUTION
        
        if agent_input.code_snippet and not agent_input.error_message:
            return DebuggingStrategy.CODE_INSPECTION
        
        # Default strategy
        return DebuggingStrategy.SYSTEMATIC_TESTING
    
    def _assess_debugging_complexity(self, agent_input: AgentInput) -> DebuggingComplexity:
        """
        Assess the complexity of the debugging task.
        
        Args:
            agent_input: Student input
            
        Returns:
            Debugging complexity level
        """
        complexity_score = 0
        
        # Factors that increase complexity
        if agent_input.code_snippet and len(agent_input.code_snippet) > 200:
            complexity_score += 1
        
        if agent_input.error_message and len(agent_input.error_message.split('\n')) > 3:
            complexity_score += 1
        
        # Multiple error types or vague descriptions
        error_keywords = ["error", "issue", "problem", "bug", "wrong", "broken"]
        query_lower = agent_input.query.lower()
        error_count = sum(1 for keyword in error_keywords if keyword in query_lower)
        if error_count > 2:
            complexity_score += 1
        
        # Vague problem descriptions
        vague_indicators = ["doesn't work", "not working", "something wrong", "broken"]
        if any(indicator in query_lower for indicator in vague_indicators):
            complexity_score += 1
        
        # Performance or design issues are typically more complex
        complex_issues = ["performance", "slow", "optimization", "design", "architecture"]
        if any(issue in query_lower for issue in complex_issues):
            complexity_score += 2
        
        if complexity_score >= 3:
            return DebuggingComplexity.COMPLEX
        elif complexity_score >= 1:
            return DebuggingComplexity.MODERATE
        else:
            return DebuggingComplexity.SIMPLE
    
    def _analyze_code_snippet(self, code: str) -> Dict[str, Any]:
        """
        Analyze code snippet for potential issues and patterns.
        
        Args:
            code: Code snippet to analyze
            
        Returns:
            Dictionary with code analysis results
        """
        if not code:
            return {}
        
        analysis = {
            "code_length": len(code),
            "line_count": len(code.split('\n')),
            "has_loops": False,
            "has_functions": False,
            "has_classes": False,
            "potential_issues": [],
            "complexity_indicators": []
        }
        
        code_lower = code.lower()
        
        # Detect code structures
        if any(keyword in code_lower for keyword in ['for ', 'while ']):
            analysis["has_loops"] = True
        
        if any(keyword in code_lower for keyword in ['def ', 'function']):
            analysis["has_functions"] = True
        
        if 'class ' in code_lower:
            analysis["has_classes"] = True
        
        # Look for potential issues
        potential_issues = []
        
        # Index-related issues
        if '[i+1]' in code or '[i-1]' in code:
            potential_issues.append("potential_index_bounds_issue")
        
        # Loop condition issues
        if 'while true' in code_lower and 'break' not in code_lower:
            potential_issues.append("potential_infinite_loop")
        
        # Uninitialized variables (simple heuristic)
        lines = code.split('\n')
        variables_used = set()
        variables_defined = set()
        
        for line in lines:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                # Variable assignment
                var_name = line.split('=')[0].strip()
                if var_name.isidentifier():
                    variables_defined.add(var_name)
            
            # Simple variable usage detection
            import re
            for match in re.finditer(r'\b([a-zA-Z_]\w*)\b', line):
                var_name = match.group(1)
                if var_name not in ['for', 'while', 'if', 'else', 'def', 'class', 'return', 'print']:
                    variables_used.add(var_name)
        
        undefined_vars = variables_used - variables_defined
        if undefined_vars:
            potential_issues.append("potential_undefined_variables")
        
        analysis["potential_issues"] = potential_issues
        
        # Complexity indicators
        complexity_indicators = []
        if analysis["line_count"] > 20:
            complexity_indicators.append("long_code")
        if analysis["has_loops"] and analysis["has_functions"]:
            complexity_indicators.append("multiple_constructs")
        if len(potential_issues) > 1:
            complexity_indicators.append("multiple_potential_issues")
        
        analysis["complexity_indicators"] = complexity_indicators
        
        return analysis
    
    def _assess_debugging_response_quality(self, response: str, agent_input: AgentInput) -> float:
        """
        Assess the quality of debugging response.
        
        Args:
            response: Generated response
            agent_input: Original input
            
        Returns:
            Quality score from 0.0 to 1.0
        """
        quality_score = 0.5  # Base score
        response_lower = response.lower()
        
        # Quality indicators for debugging responses
        debugging_quality_indicators = [
            "understand", "analyze", "check", "verify", "test",
            "step by step", "systematic", "method", "approach"
        ]
        
        guidance_indicators = [
            "consider", "try", "look at", "examine", "investigate",
            "what happens", "how does", "why might"
        ]
        
        educational_indicators = [
            "learn", "understand", "practice", "skill", "strategy",
            "debugging", "approach", "method"
        ]
        
        # Bonus for debugging methodology
        quality_score += min(0.2, sum(0.03 for ind in debugging_quality_indicators if ind in response_lower))
        
        # Bonus for guidance approach
        quality_score += min(0.15, sum(0.03 for ind in guidance_indicators if ind in response_lower))
        
        # Bonus for educational focus
        quality_score += min(0.15, sum(0.03 for ind in educational_indicators if ind in response_lower))
        
        # Penalty for direct solutions without explanation
        direct_solution_indicators = [
            "here's the fix", "change this to", "replace with",
            "just add", "simply change"
        ]
        
        quality_score -= min(0.2, sum(0.05 for ind in direct_solution_indicators if ind in response_lower))
        
        # Bonus for addressing specific error
        if agent_input.error_message:
            error_terms = agent_input.error_message.lower().split()[:3]  # First few words
            if any(term in response_lower for term in error_terms):
                quality_score += 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def _assess_debugging_educational_value(self, response: str) -> float:
        """
        Assess educational value of debugging response.
        
        Args:
            response: Generated response
            
        Returns:
            Educational value score from 0.0 to 1.0
        """
        value_score = 0.0
        response_lower = response.lower()
        
        # Educational value indicators
        skill_building = [
            "debugging skill", "systematic approach", "method",
            "strategy", "technique", "process"
        ]
        
        understanding = [
            "understand why", "learn from", "recognize",
            "pattern", "common mistake", "typical error"
        ]
        
        metacognitive = [
            "think about", "consider", "reflect",
            "approach", "strategy", "method"
        ]
        
        # Score for different educational aspects
        for indicator in skill_building:
            if indicator in response_lower:
                value_score += 0.2
        
        for indicator in understanding:
            if indicator in response_lower:
                value_score += 0.15
        
        for indicator in metacognitive:
            if indicator in response_lower:
                value_score += 0.1
        
        # Bonus for systematic approach
        systematic_indicators = [
            "step 1", "first", "next", "then", "finally",
            "systematic", "methodical", "organized"
        ]
        
        if any(ind in response_lower for ind in systematic_indicators):
            value_score += 0.2
        
        return min(1.0, value_score)
    
    def _identify_debugging_concepts(self, agent_input: AgentInput, response: str) -> List[str]:
        """
        Identify debugging concepts covered in the interaction.
        
        Args:
            agent_input: Student input
            response: Generated response
            
        Returns:
            List of debugging concepts covered
        """
        concepts = []
        combined_text = (agent_input.query + " " + response).lower()
        
        # Debugging skill concepts
        debugging_concepts = {
            "error_analysis": ["error message", "error analysis", "interpreting errors"],
            "systematic_debugging": ["systematic", "step by step", "methodical"],
            "code_inspection": ["code review", "inspection", "line by line"],
            "testing_strategies": ["testing", "test cases", "validation"],
            "trace_execution": ["trace", "execution", "step through"],
            "hypothesis_testing": ["hypothesis", "theory", "assumption"],
            "isolation": ["isolate", "narrow down", "minimal example"],
            "pattern_recognition": ["pattern", "common error", "typical mistake"]
        }
        
        for concept, indicators in debugging_concepts.items():
            if any(indicator in combined_text for indicator in indicators):
                concepts.append(concept)
        
        # Error type concepts
        error_categories = [
            "syntax_errors", "runtime_errors", "logic_errors",
            "type_errors", "index_errors", "name_errors"
        ]
        
        for category in error_categories:
            category_name = category.replace("_", " ")
            if category_name in combined_text:
                concepts.append(category)
        
        return concepts
    
    def _generate_debugging_learning_objectives(self, 
                                              strategy: DebuggingStrategy,
                                              error_category: ErrorCategory) -> List[str]:
        """
        Generate learning objectives for debugging session.
        
        Args:
            strategy: Debugging strategy used
            error_category: Type of error encountered
            
        Returns:
            List of learning objectives
        """
        objectives = []
        
        # Strategy-based objectives
        strategy_objectives = {
            DebuggingStrategy.ERROR_MESSAGE_ANALYSIS: [
                "Learn to interpret error messages systematically",
                "Identify the root cause from error information",
                "Understand common error message patterns"
            ],
            DebuggingStrategy.CODE_INSPECTION: [
                "Develop systematic code review skills",
                "Identify potential issues through static analysis",
                "Build pattern recognition for common bugs"
            ],
            DebuggingStrategy.SYSTEMATIC_TESTING: [
                "Apply systematic testing methodologies",
                "Design effective test cases for debugging",
                "Use testing to isolate and verify fixes"
            ],
            DebuggingStrategy.LOGICAL_REASONING: [
                "Apply logical reasoning to debug problems",
                "Trace program execution mentally",
                "Identify logical flaws in algorithm design"
            ]
        }
        
        objectives.extend(strategy_objectives.get(strategy, []))
        
        # Error-category-based objectives
        if error_category == ErrorCategory.INDEX_ERROR:
            objectives.append("Understand array bounds and index management")
        elif error_category == ErrorCategory.TYPE_ERROR:
            objectives.append("Master data type compatibility and conversion")
        elif error_category == ErrorCategory.LOGIC_ERROR:
            objectives.append("Develop algorithm verification and testing skills")
        
        return objectives[:4]  # Limit to top 4 objectives
    
    def _generate_debugging_follow_up(self, 
                                    agent_input: AgentInput,
                                    strategy: DebuggingStrategy,
                                    error_category: ErrorCategory) -> List[str]:
        """
        Generate debugging-specific follow-up questions.
        
        Args:
            agent_input: Original input
            strategy: Debugging strategy used
            error_category: Error category
            
        Returns:
            List of follow-up questions
        """
        questions = []
        
        # Strategy-specific questions
        if strategy == DebuggingStrategy.ERROR_MESSAGE_ANALYSIS:
            questions.extend([
                "What specific information does the error message provide?",
                "Can you trace the error back to its source?",
                "What line number or component is mentioned in the error?"
            ])
        elif strategy == DebuggingStrategy.CODE_INSPECTION:
            questions.extend([
                "Can you walk through your code line by line?",
                "What do you expect each part of the code to do?",
                "Are there any parts that look suspicious or unclear?"
            ])
        elif strategy == DebuggingStrategy.SYSTEMATIC_TESTING:
            questions.extend([
                "What's the simplest test case you can create?",
                "Can you reproduce the problem consistently?",
                "What happens when you test with different inputs?"
            ])
        
        # Error-specific questions
        if error_category == ErrorCategory.INDEX_ERROR:
            questions.append("What are the valid index ranges for your data structure?")
        elif error_category == ErrorCategory.TYPE_ERROR:
            questions.append("What data types are involved in the operation that's failing?")
        elif error_category == ErrorCategory.LOGIC_ERROR:
            questions.append("Can you trace through your algorithm with a simple example?")
        
        # General debugging questions
        questions.extend([
            "How will you test your fix to make sure it works?",
            "What did you learn from this debugging process?",
            "How could you prevent similar errors in the future?"
        ])
        
        return questions[:5]  # Limit to top 5 questions
    
    def _uses_systematic_approach(self, response: str) -> bool:
        """Check if response promotes systematic debugging."""
        systematic_indicators = [
            "step", "systematic", "methodical", "organized",
            "first", "next", "then", "process", "approach"
        ]
        response_lower = response.lower()
        return sum(1 for ind in systematic_indicators if ind in response_lower) >= 2
    
    def _teaches_debugging_skills(self, response: str) -> bool:
        """Check if response teaches general debugging skills."""
        skill_indicators = [
            "debugging", "debug", "skill", "technique", "method",
            "strategy", "approach", "process", "learn"
        ]
        response_lower = response.lower()
        return any(ind in response_lower for ind in skill_indicators)
    
    def _encourages_testing(self, response: str) -> bool:
        """Check if response encourages testing and validation."""
        testing_indicators = [
            "test", "testing", "validate", "verify", "check",
            "try", "experiment", "reproduce"
        ]
        response_lower = response.lower()
        return any(ind in response_lower for ind in testing_indicators)
    
    def _builds_error_understanding(self, response: str) -> bool:
        """Check if response builds understanding of errors."""
        understanding_indicators = [
            "understand", "why", "because", "reason", "cause",
            "explain", "error message", "meaning"
        ]
        response_lower = response.lower()
        return any(ind in response_lower for ind in understanding_indicators)
    
    def _determine_debugging_response_type(self, 
                                         strategy: DebuggingStrategy,
                                         error_category: ErrorCategory) -> ResponseType:
        """Determine appropriate response type for debugging context."""
        if strategy in [DebuggingStrategy.ERROR_MESSAGE_ANALYSIS, DebuggingStrategy.CODE_INSPECTION]:
            return ResponseType.EXPLANATION
        elif strategy in [DebuggingStrategy.SYSTEMATIC_TESTING, DebuggingStrategy.TRACE_EXECUTION]:
            return ResponseType.DEBUGGING_HELP
        else:
            return ResponseType.DEBUGGING_HELP


if __name__ == "__main__":
    # Debugging agent test
    try:
        agent = DebuggingAgent()
        
        # Test cases for different debugging scenarios
        test_cases = [
            AgentInput(
                query="I'm getting an IndexError: list index out of range",
                code_snippet="for i in range(len(arr)):\n    if arr[i] > arr[i+1]:",
                error_message="IndexError: list index out of range",
                student_level="beginner"
            ),
            AgentInput(
                query="My binary search function returns the wrong result",
                code_snippet="def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                student_level="intermediate"
            ),
            AgentInput(
                query="Why does my recursive function cause a stack overflow?",
                code_snippet="def factorial(n):\n    return n * factorial(n-1)",
                error_message="RecursionError: maximum recursion depth exceeded",
                student_level="intermediate"
            )
        ]
        
        for i, test_input in enumerate(test_cases):
            print(f"\nTest Case {i+1}: {test_input.query}")
            
            response = agent.process_query(test_input)
            
            print(f"Response Type: {response.response_type}")
            print(f"Confidence: {response.confidence:.3f}")
            print(f"Strategy: {response.educational_metadata.get('debugging_strategy')}")
            print(f"Error Category: {response.educational_metadata.get('error_category')}")
            print(f"Complexity: {response.educational_metadata.get('debugging_complexity')}")
            print(f"Concepts: {response.educational_metadata.get('concepts_covered', [])}")
            print(f"Follow-up Questions: {len(response.suggested_follow_up or [])}")
        
        # Performance stats
        stats = agent.get_performance_stats()
        print(f"\nAgent Performance: {stats}")
        
        print("✅ Debugging agent test completed successfully!")
        
    except Exception as e:
        print(f"❌ Debugging agent test failed: {e}")
        import traceback
        traceback.print_exc()
