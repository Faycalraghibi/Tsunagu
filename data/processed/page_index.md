# Page index

Use these page ids in `gold_pages`. Headings are listed to help write questions.

## langgraph/overview: LangGraph overview
1336 tokens, links to: langgraph/persistence, langgraph/interrupts, concepts/memory
- <Icon icon="download" size={20} /> Install
- Core benefits
- LangGraph ecosystem
- Acknowledgements

## langgraph/install: Install LangGraph
170 tokens, links to: none

## langgraph/quickstart: Quickstart
4165 tokens, links to: langgraph/graph-api, langgraph/functional-api
- Step 1: Read the guide
- Step 2: Install dependencies
- Step 3: Configure model credentials
- Step 4: Implement the calculator agent
- Rules
- 1. Define tools and model
- 2. Define state
- 3. Define model node
- 4. Define tool node
- 5. Define end logic
- 6. Build and compile the agent
- 2. Define model node
- 3. Define tool node
- 4. Define agent

## langgraph/thinking-in-langgraph: Thinking in LangGraph
5337 tokens, links to: concepts/memory, langgraph/fault-tolerance, langgraph/interrupts, langgraph/persistence, langgraph/checkpointers
- Start with the process you want to automate
- Step 1: Map out your workflow as discrete steps
- Step 2: Identify what each step needs to do
- Step 2: Identify what each step needs to do > LLM steps
- Step 2: Identify what each step needs to do > Data steps
- Step 2: Identify what each step needs to do > Action steps
- Step 2: Identify what each step needs to do > User input steps
- Step 3: Design your state
- Step 3: Design your state > What belongs in state?
- Step 3: Design your state > Keep state raw, format prompts on-demand
- Step 4: Build your nodes
- Step 4: Build your nodes > Handle errors appropriately
- Step 4: Build your nodes > Implementing our email agent nodes
- Step 5: Wire it together
- Step 5: Wire it together > Try out your agent
- Summary and next steps > Key Insights
- Summary and next steps > Advanced considerations
- Summary and next steps > Where to go from here

## langgraph/choosing-apis: Choosing between Graph and Functional APIs
2180 tokens, links to: langgraph/graph-api, langgraph/functional-api
- Quick decision guide
- Detailed comparison > When to use the Graph API
- Detailed comparison > When to use the Functional API
- Combining both APIs
- Migration between APIs > From Functional to Graph API
- Migration between APIs > From Graph to Functional API
- Summary

## langgraph/graph-api: Graph API overview
10271 tokens, links to: langgraph/persistence, langgraph/use-graph-api, langgraph/fault-tolerance, langgraph/interrupts, langgraph/functional-api, langgraph/use-subgraphs, langchain/multi-agent/handoffs, langgraph/backward-compatibility, langgraph/choosing-apis
- Graphs
- Graphs > StateGraph
- Graphs > Compiling your graph
- State
- State > Schema
- State > Schema > Multiple schemas
- State > Reducers
- State > Reducers > Reducer arguments
- State > Reducers > Default reducer
- State > Reducers > Custom reducers
- State > Reducers > Overwrite
- State > Reducers > Resetting a reducer field
- State > Untracked values
- State > Type utilities
- State > Type utilities > `GraphNode`
- State > Type utilities > `State.Node` shorthand
- State > Type utilities > `ConditionalEdgeRouter`
- State > Type utilities > `StateSchema.State` and `StateSchema.Update`
- State > Working with messages in graph state > Why use messages?
- State > Working with messages in graph state > Using messages in your graph
- State > Working with messages in graph state > Serialization
- State > Working with messages in graph state > MessagesState
- Nodes
- Nodes > Re-execution and idempotency
- Nodes > Using tasks in nodes
- Nodes > `START` node
- Nodes > `END` node
- Nodes > Node caching
- Edges
- Edges > Normal edges
- Edges > Conditional edges
- Edges > Entry point
- Edges > Conditional entry point
- `Send`
- `Command`
- `Command` > Return from nodes > `update` and `goto`
- `Command` > Return from nodes > `graph`
- `Command` > Input to `invoke` or `stream`
- `Command` > Input to `invoke` or `stream` > `resume`
- `Command` > Return from tools
- Graph migrations
- Runtime context
- Runtime context > Recursion limit
- Runtime context > Accessing and handling the recursion counter
- Runtime context > Accessing and handling the recursion counter > How it works
- Runtime context > Accessing and handling the recursion counter > Accessing the current step counter
- Runtime context > Accessing and handling the recursion counter > Proactive recursion handling
- Runtime context > Accessing and handling the recursion counter > Proactive vs reactive approaches
- Runtime context > Accessing and handling the recursion counter > Other available metadata
- Visualization
- Observability and Tracing
- Learn more

## langgraph/use-graph-api: Use the graph API
18839 tokens, links to: langgraph/graph-api, langgraph/fault-tolerance, langgraph/persistence, langgraph/interrupts, langgraph/use-time-travel, langgraph/streaming, langgraph/workflows-agents, langgraph/use-subgraphs
- Setup
- Define and update state
- Define and update state > Define state
- Define and update state > Update state
- Define and update state > Process state updates with reducers
- Define and update state > Process state updates with reducers > MessagesState
- Define and update state > Bypass reducers with `Overwrite`
- Define and update state > Define input and output schemas
- Define and update state > Pass private state between nodes
- Define and update state > Use pydantic models for graph state
- Add runtime configuration
- Add retry policies
- Set node timeouts
- Configure node timeouts
- Handle node errors
- Set graph-wide node defaults
- Set graph-wide node defaults > Access execution info inside a node
- Set graph-wide node defaults > Access execution info inside a node > Access thread and run IDs
- Set graph-wide node defaults > Access execution info inside a node > Adjust behavior based on retry state
- Set graph-wide node defaults > Access server info inside a node
- Set graph-wide node defaults > Access drain state inside a node
- Add node caching
- Create a sequence of steps
- Create branches
- Create branches > Run graph nodes in parallel
- Create branches > Defer node execution
- Create branches > Conditional branching
- Map-Reduce and the send API
- Create and control loops
- Create and control loops > Impose a recursion limit
- Async
- Combine control flow and state updates with `Command`
- Combine control flow and state updates with `Command` > Navigate to a node in a parent graph
- Combine control flow and state updates with `Command` > Use inside tools
- Visualize your graph
- Visualize your graph > Mermaid
- Visualize your graph > PNG

## langgraph/functional-api: Functional API overview
5423 tokens, links to: langgraph/persistence, langgraph/add-memory, langgraph/interrupts, langgraph/streaming, langgraph/use-functional-api, langgraph/graph-api, langgraph/checkpointers, langgraph/use-subgraphs, langgraph/choosing-apis
- Functional API vs. Graph API
- Example
- Entrypoint
- Entrypoint > Definition
- Entrypoint > Injectable parameters
- Entrypoint > Executing
- Entrypoint > Resuming
- Entrypoint > Short-term memory
- Entrypoint > Short-term memory > `entrypoint.final`
- Task
- Task > Definition
- Task > Execution
- When to use a task
- Serialization
- Determinism
- Idempotency
- Common pitfalls > Handling side effects
- Common pitfalls > Non-deterministic control flow
- Learn more

## langgraph/use-functional-api: Use the functional API
7157 tokens, links to: langgraph/functional-api, langgraph/persistence, langgraph/add-memory, langgraph/interrupts, langgraph/streaming, langgraph/graph-api, concepts/memory, langgraph/workflows-agents
- Creating a simple workflow
- Parallel execution
- Calling graphs
- Call other entrypoints
- Streaming
- Retry policy
- Set task and entrypoint timeouts
- Caching tasks
- Resuming after an error
- Human-in-the-loop
- Human-in-the-loop > Basic human-in-the-loop workflow
- Human-in-the-loop > Review tool calls
- Short-term memory
- Short-term memory > Manage checkpoints
- Short-term memory > Manage checkpoints > View thread state
- Short-term memory > Manage checkpoints > View the history of the thread
- Short-term memory > Decouple return value from saved value
- Short-term memory > Chatbot example
- Long-term memory
- Workflows
- Integrate with other libraries

## langgraph/pregel: LangGraph runtime
4262 tokens, links to: langgraph/graph-api, langgraph/functional-api
- Overview
- Actors
- Channels
- Channels > LastValue
- Channels > Topic
- Channels > BinaryOperatorAggregate
- Channels > DeltaChannel
- Channels > DeltaChannel > Bulk reducer requirement
- Channels > DeltaChannel > Use snapshot_frequency for bounded read latency
- Channels > DeltaChannel > Version compatibility and rollbacks
- Examples
- High-level API

## langgraph/persistence: Persistence
908 tokens, links to: langgraph/checkpointers, langgraph/stores
- Quickstart
- Checkpointer vs. store
- Troubleshooting common issues > PostgresSaver: `thread_id` too long
- Troubleshooting common issues > `MemorySaver` does not persist between restarts
- Troubleshooting common issues > Checkpoints growing unboundedly
- Troubleshooting common issues > State access from parent graph to subgraph
- Next steps

## langgraph/checkpointers: Checkpointers
9084 tokens, links to: langgraph/interrupts, concepts/memory, langgraph/add-memory, langgraph/use-time-travel, langgraph/graph-api, langgraph/use-subgraphs, langgraph/pregel
- Why use checkpointers
- Core concepts > Threads
- Core concepts > Checkpoints
- Core concepts > Checkpoints > Super-steps
- Core concepts > Checkpoints > Checkpoint namespace
- Get and update state > Get state
- Get and update state > Get state > StateSnapshot fields
- Get and update state > Get state history
- Get and update state > Get state history > Find a specific checkpoint
- Get and update state > Replay
- Get and update state > Update state
- Durability modes
- Optimize checkpoint storage
- Checkpointer libraries
- Checkpointer libraries > Checkpointer interface
- Checkpointer libraries > Serializer
- Checkpointer libraries > Serializer > Serialization with `pickle`
- Checkpointer libraries > Serializer > Encryption
- Build a custom checkpointer
- Build a custom checkpointer > Overview
- Build a custom checkpointer > Base contract
- Build a custom checkpointer > Base contract > put / aput
- Build a custom checkpointer > Base contract > put_writes / aput_writes
- Build a custom checkpointer > Base contract > get_tuple / aget_tuple
- Build a custom checkpointer > Base contract > list / alist
- Build a custom checkpointer > Base contract > delete_thread / adelete_thread
- Build a custom checkpointer > Row key / index design
- Build a custom checkpointer > Serialization
- Build a custom checkpointer > Extended capabilities
- Build a custom checkpointer > Delta channel support
- Build a custom checkpointer > Delta channel support > What the runtime needs
- Build a custom checkpointer > Delta channel support > Default implementation
- Build a custom checkpointer > Delta channel support > Performance override
- Build a custom checkpointer > Delta channel support > Pruning with delta channels
- Build a custom checkpointer > Delta channel support > Copy thread with delta channels
- Build a custom checkpointer > Testing with the conformance suite

## langgraph/stores: Stores
3425 tokens, links to: langgraph/checkpointers
- Basic usage
- Listing items in a namespace
- Semantic search
- Using in LangGraph
- Build a custom store
- Build a custom store > Base contract
- Build a custom store > Namespace design
- Build a custom store > Serialization
- Build a custom store > Semantic search support
- Build a custom store > Testing
- Build a custom store > Next steps

## langgraph/add-memory: Memory
16853 tokens, links to: concepts/memory, langgraph/graph-api, langgraph/persistence, langgraph/use-subgraphs
- Add short-term memory
- Add short-term memory > Use in production
- Add short-term memory > Use in subgraphs
- Add long-term memory
- Add long-term memory > Access the store inside nodes
- Add long-term memory > Use in production
- Add long-term memory > Use semantic search
- Manage short-term memory
- Manage short-term memory > Trim messages
- Manage short-term memory > Delete messages
- Manage short-term memory > Summarize messages
- Manage short-term memory > Manage checkpoints
- Manage short-term memory > Manage checkpoints > View thread state
- Manage short-term memory > Manage checkpoints > View the history of the thread
- Manage short-term memory > Manage checkpoints > Delete all checkpoints for a thread
- Database management

## langgraph/interrupts: Interrupts
6890 tokens, links to: langgraph/persistence, langgraph/event-streaming, langgraph/graph-api
- Pause using `interrupt`
- Resuming interrupts
- Define an interrupt response schema
- Common patterns
- Common patterns > Stream with human-in-the-loop (HITL) interrupts
- Common patterns > Handling multiple interrupts
- Common patterns > Approve or reject
- Common patterns > Review and edit state
- Common patterns > Interrupts in tools
- Common patterns > Validating human input
- Rules of interrupts
- Rules of interrupts > Do not wrap `interrupt` calls in try/except
- Rules of interrupts > Do not reorder `interrupt` calls within a node
- Rules of interrupts > Do not return complex values in `interrupt` calls
- Rules of interrupts > Side effects called before `interrupt` must be idempotent
- Using with subgraphs called as functions
- Debugging with interrupts
- Debugging with interrupts > Using LangSmith Studio

## langgraph/use-time-travel: Use time-travel
2472 tokens, links to: langgraph/checkpointers, langgraph/interrupts, langgraph/graph-api, langgraph/test, langgraph/use-subgraphs
- Overview
- Replay
- Fork
- Fork > From a specific node
- Interrupts
- Interrupts > Multiple interrupts
- Subgraphs

## langgraph/fault-tolerance: Fault tolerance
5595 tokens, links to: none
- Retries
- Retries > Default behavior
- Retries > Parameters
- Retries > Custom retry logic
- Retries > Inspect retry state
- Timeouts
- Timeouts > Run timeout
- Timeouts > Idle timeout
- Timeouts > Idle timeout > Progress signals
- Timeouts > Idle timeout > Heartbeat mode
- Timeouts > Idle timeout > Manual heartbeats
- Timeouts > NodeTimeoutError
- Timeouts > Dynamic timeouts with Send
- Error handling
- Error handling > NodeError
- Error handling > Route with Command
- Error handling > Resume-safe failures
- Error handling > Behavior with `interrupt()`
- Error handling > Subgraph failures
- Graph defaults
- Graph defaults > Precedence
- Graph defaults > Default error handler
- Graph defaults > Applicability matrix
- Graph defaults > Scope
- Functional API
- Graceful shutdown
- Graceful shutdown > Semantics
- Graceful shutdown > Resume after drain
- Graceful shutdown > Read drain state inside a node
- Graceful shutdown > SIGTERM hook pattern
- Limitations

## langgraph/streaming: Streaming
9416 tokens, links to: langgraph/event-streaming, langgraph/use-subgraphs, langgraph/persistence
- Get started > Basic usage
- Get started > Stream output format (v2)
- Stream modes
- Stream modes > Graph state
- Stream modes > LLM tokens
- Stream modes > LLM tokens > Filter by LLM invocation
- Stream modes > LLM tokens > Omit messages from the stream
- Stream modes > LLM tokens > Filter by node
- Stream modes > Custom data
- Stream modes > Subgraph outputs
- Stream modes > Checkpoints
- Stream modes > Tasks
- Stream modes > Debug
- Stream modes > Multiple modes at once
- Advanced > Use with any LLM
- Advanced > Disable streaming for specific chat models
- Advanced > Migrate to v2
- Advanced > Migrate to v2 > v2 invoke format
- Advanced > Migrate to v2 > Pydantic and dataclass state coercion
- Advanced > Async with Python < 3.11

## langgraph/event-streaming: Event streaming
5283 tokens, links to: langgraph/streaming, langgraph/persistence, langgraph/frontend/overview
- Quickstart
- How the pieces fit together
- What event streaming provides
- Stream messages
- Stream subgraphs
- Stream state
- Stream multiple projections
- Resume after an interrupt
- Stream all protocol events
- Channels and event lifecycle
- Channels and event lifecycle > Messages
- Channels and event lifecycle > Tools
- Channels and event lifecycle > Lifecycle
- Build your own projection
- Build your own projection > How transformers work
- Build your own projection > Transformer shape
- Build your own projection > Declaring required stream modes
- Build your own projection > StreamChannel
- Build your own projection > Example: named channel
- Build your own projection > Example: unnamed channel
- Build your own projection > Example: final-value projection
- Build your own projection > Register at call time or compile time
- Build your own projection > Built-in: `ToolCallTransformer`
- Related

## langgraph/use-subgraphs: Subgraphs
7903 tokens, links to: langgraph/graph-api, langchain/multi-agent, langgraph/interrupts, langgraph/persistence, langgraph/checkpointers, langgraph/event-streaming
- Setup
- Define subgraph communication
- Define subgraph communication > Call a subgraph inside a node
- Define subgraph communication > Add a subgraph as a node
- Subgraph persistence
- Subgraph persistence > Stateful
- Subgraph persistence > Stateful > Per-invocation (default)
- Subgraph persistence > Stateful > Per-thread
- Subgraph persistence > Stateless
- Subgraph persistence > Checkpointer reference
- View subgraph state
- Stream subgraph outputs

## langgraph/workflows-agents: Workflows and agents
7826 tokens, links to: langgraph/persistence, langgraph/streaming, langgraph/deploy, langgraph/interrupts
- Setup
- LLMs and augmentations
- Prompt chaining
- Parallelization
- Routing
- Orchestrator-worker
- Orchestrator-worker > Creating workers in LangGraph
- Evaluator-optimizer
- Agents
- Agents > ToolNode
- Agents > ToolNode > Access graph state and context from tools

## langgraph/agentic-rag: Build a custom RAG agent with LangGraph
1810 tokens, links to: langgraph/overview, langgraph/graph-api
- Concepts
- Setup
- Setup > Set up LangSmith
- Preprocess documents
- Create a retriever tool
- Generate a query or respond
- Grade documents
- Rewrite the question
- Generate an answer
- Assemble the graph
- Run the agentic RAG
- See also

## langgraph/sql-agent: Build a custom SQL agent
2706 tokens, links to: langgraph/overview, langgraph/graph-api, langgraph/interrupts, langgraph/persistence, langgraph/use-graph-api
- Concepts
- Setup > Installation
- Setup > LangSmith
- 1. Select an LLM
- 2. Configure the database
- 3. Add tools for database interactions
- 4. Define application steps
- 5. Implement the agent
- 6. Implement human-in-the-loop review
- Next steps

## langgraph/application-structure: Application structure
1066 tokens, links to: none
- Key concepts
- File structure
- Configuration file
- Configuration file > Examples
- Dependencies
- Graphs
- Environment variables

## langgraph/test: Test
1395 tokens, links to: langgraph/use-subgraphs, langgraph/use-time-travel
- Prerequisites
- Getting started
- Testing individual nodes and edges
- Partial execution

## langgraph/local-server: Run a local server
1463 tokens, links to: none
- Prerequisites
- 1. Install the LangGraph CLI
- 2. Create a LangGraph app
- 3. Install dependencies
- 4. Create a `.env` file
- 5. Launch Agent server
- 6. Test your application in Studio
- 7. Test the API
- Next steps

## langgraph/deploy: Deployment
854 tokens, links to: langgraph/studio
- LangSmith Cloud
- LangSmith Cloud > Prerequisites
- LangSmith Cloud > Deploy your agent > 1. Create a repository on GitHub
- LangSmith Cloud > Deploy your agent > 2. Deploy to LangSmith
- LangSmith Cloud > Deploy your agent > 3. Test your application in Studio
- LangSmith Cloud > Deploy your agent > 4. Get the API URL for your deployment
- LangSmith Cloud > Deploy your agent > 5. Test the API

## langgraph/studio: LangSmith Studio
1355 tokens, links to: none
- Prerequisites
- Set up local Agent server > 1. Install the LangGraph CLI
- Set up local Agent server > 2. Prepare your agent
- Set up local Agent server > 3. Environment variables
- Set up local Agent server > 4. Create a LangGraph config file
- Set up local Agent server > 5. Install dependencies
- Set up local Agent server > 6. View your agent in Studio
- Video guide

## langgraph/observability: LangSmith Observability
1039 tokens, links to: none
- Prerequisites
- Enable tracing
- Trace selectively
- Log to a project
- Add metadata to traces
- Use anonymizers to prevent logging of sensitive data in traces

## langgraph/ui: Agent Chat UI
234 tokens, links to: langgraph/studio, langgraph/deploy
- Connect to your agent

## langgraph/backward-compatibility: Backward compatibility
2223 tokens, links to: langgraph/persistence, langgraph/functional-api, langgraph/graph-api, langgraph/checkpointers, langgraph/use-time-travel, langgraph/observability
- Technical compatibility
- Technical compatibility > Recommended patterns
- Technical compatibility > Detecting in-flight threads
- Business compatibility
- Non-determinism

## langgraph/errors/GRAPH_RECURSION_LIMIT: GRAPH_RECURSION_LIMIT
204 tokens, links to: none
- Troubleshooting

## langgraph/errors/INVALID_CHAT_HISTORY: INVALID_CHAT_HISTORY
452 tokens, links to: none
- Troubleshooting

## langgraph/errors/INVALID_CONCURRENT_GRAPH_UPDATE: INVALID_CONCURRENT_GRAPH_UPDATE
395 tokens, links to: langgraph/use-graph-api
- Troubleshooting

## langgraph/errors/INVALID_GRAPH_NODE_RETURN_VALUE: INVALID_GRAPH_NODE_RETURN_VALUE
229 tokens, links to: none
- Troubleshooting

## langgraph/errors/MISSING_CHECKPOINTER: MISSING_CHECKPOINTER
211 tokens, links to: langgraph/persistence
- Troubleshooting
- Related

## langgraph/errors/MULTIPLE_SUBGRAPHS: MULTIPLE_SUBGRAPHS
304 tokens, links to: langgraph/use-subgraphs, langgraph/persistence
- Troubleshooting
- Related

## langgraph/frontend/overview: Overview
1168 tokens, links to: none
- Architecture
- What makes this different from a chat stream
- Patterns
- Related patterns

## langgraph/frontend/graph-execution: Graph execution
2980 tokens, links to: none
- How graph nodes map to UI cards
- Setting up `useStream`
- Routing streaming tokens to nodes
- Determining node status
- Building the pipeline progress bar
- Building collapsible NodeCard components
- Streaming vs. completed content
- Putting it all together
- Use cases
- Handling dynamic pipelines
- Best practices

## langgraph/frontend/custom-stream-channels: Custom stream channels
2394 tokens, links to: langgraph/frontend/graph-execution, langgraph/frontend/overview
- How custom channels work
- Setting up `useStream`
- Read the latest payload with `useExtension`
- Buffer raw events with `useChannel`
- Choosing between `useExtension` and `useChannel`
- Use cases
- Related

## concepts/memory: Memory overview
3713 tokens, links to: langgraph/add-memory, langgraph/checkpointers, langgraph/graph-api, langgraph/stores, concepts/context
- Short-term memory
- Short-term memory > Manage short-term memory
- Long-term memory
- Long-term memory > Semantic memory
- Long-term memory > Semantic memory > Profile
- Long-term memory > Semantic memory > Collection
- Long-term memory > Episodic memory
- Long-term memory > Procedural memory
- Long-term memory > Writing memories
- Long-term memory > Writing memories > In the hot path
- Long-term memory > Writing memories > In the background
- Long-term memory > Memory storage
- Learn more

## concepts/human-in-the-loop: Human-in-the-loop overview
1856 tokens, links to: langchain/human-in-the-loop, langgraph/persistence, langgraph/interrupts
- Follow an approval from pause to resume
- Review each tool call separately
- Review a whole turn at once
- Compare batched-mode problems
- Choose an interrupt mode
- See also

## concepts/context: Context overview
1691 tokens, links to: langgraph/use-graph-api, concepts/memory, langgraph/add-memory
- Static runtime context
- Dynamic runtime context
- Dynamic cross-conversation context
- Learn more

## langchain/human-in-the-loop: Human-in-the-loop
5303 tokens, links to: langgraph/persistence, langgraph/interrupts, concepts/human-in-the-loop
- Interrupt decision types
- Configuring interrupts
- Conditional interrupts
- Set the interrupt mode
- Pause once per tool call
- Pause once per tool call > Read the interrupts
- Pause once per tool call > Read the response schema
- Pause once per tool call > Resume each call by ID
- Pause once per tool call > Correct an invalid answer
- Pause once per tool call > Order middleware for per-call review
- Responding to interrupts
- Responding to interrupts > Decision types
- Responding to interrupts > Multiple decisions
- Streaming with human-in-the-loop
- Execution lifecycle
- Custom HITL logic

## langchain/multi-agent: Multi-agent
3199 tokens, links to: langchain/multi-agent/handoffs, langchain/multi-agent/custom-workflow, langgraph/overview
- Why multi-agent?
- Patterns
- Patterns > Choosing a pattern
- Patterns > Visual overview
- Performance comparison
- Performance comparison > One-shot request
- Performance comparison > Repeat request
- Performance comparison > Multi-domain
- Performance comparison > Summary

## langchain/multi-agent/handoffs: Handoffs
3738 tokens, links to: langgraph/graph-api
- Key characteristics
- When to use
- Basic implementation
- Implementation approaches
- Implementation approaches > Single agent with middleware
- Implementation approaches > Multiple agent subgraphs
- Implementation approaches > Multiple agent subgraphs > Context engineering
- Implementation considerations

## langchain/multi-agent/custom-workflow: Custom workflow
1949 tokens, links to: langgraph/overview
- Key characteristics
- When to use
- Basic implementation
- Example: RAG pipeline
