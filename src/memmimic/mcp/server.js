#!/usr/bin/env node
/**
 * MemMimic MCP Server v1.0 - Clean Professional Edition
 * "The Memory System That Learns You Back"
 * 
 * CLEAN API: 11 essential tools, zero debug noise
 * Architecture: MCP + Python bridge + MemMimic core
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { 
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

// Setup paths
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const MEMMIMIC_DIR = __dirname;
const PYTHON_TOOLS_DIR = path.join(MEMMIMIC_DIR);

// Logging utility
function log(message) {
  const timestamp = new Date().toISOString();
  console.error(`[MEMMIMIC-MCP ${timestamp}] ${message}`);
}

// Python bridge utility - Enhanced for MemMimic
async function runPythonTool(toolName, args = []) {
  return new Promise((resolve, reject) => {
    log(`Executing: ${toolName} with args: ${JSON.stringify(args)}`);
    
    const scriptPath = path.join(PYTHON_TOOLS_DIR, `${toolName}.py`);
    
    // Check script existence
    if (!fs.existsSync(scriptPath)) {
      reject(new Error(`Tool not found: ${scriptPath}`));
      return;
    }
    
    // Use system Python 3
    const isWindows = process.platform === 'win32';
    const pythonExecutable = isWindows 
      ? 'python'
      : '/home/evilbastardxd/miniconda3/bin/python3';
    
    const pythonProcess = spawn(pythonExecutable, [scriptPath, ...args], {
      cwd: path.join(MEMMIMIC_DIR, '..', '..', '..'),
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { 
        ...process.env, 
        PYTHONPATH: path.join(MEMMIMIC_DIR, '..', '..'),
        PYTHONIOENCODING: 'utf-8',
        PYTHONUTF8: '1'
      }
    });
    
    let stdout = '';
    let stderr = '';
    
    pythonProcess.stdout.setEncoding('utf8');
    pythonProcess.stderr.setEncoding('utf8');
    
    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          // Parse JSON if possible, otherwise return raw text
          const result = stdout.trim();
          if (result.startsWith('{') || result.startsWith('[')) {
            resolve(JSON.parse(result));
          } else {
            resolve(result);
          }
        } catch (parseError) {
          resolve(stdout.trim());
        }
      } else {
        log(`Tool ${toolName} failed with code ${code}`);
        log(`STDERR: ${stderr}`);
        reject(new Error(`Tool failed: ${stderr || 'Unknown error'}`));
      }
    });
    
    pythonProcess.on('error', (error) => {
      log(`Failed to start process: ${error.message}`);
      reject(error);
    });
    
    // Timeout management
    const timeout = 30000; // 30 seconds standard
    setTimeout(() => {
      pythonProcess.kill();
      reject(new Error(`Tool ${toolName} timed out`));
    }, timeout);
  });
}

// ============================================================================
// MEMMIMIC CLEAN API - 11 ESSENTIAL TOOLS (ORIGINAL DESIGN RESTORED)
// ============================================================================

const MEMMIMIC_TOOLS = {
  // 🔍 SEARCH CORE (1)
  recall_cxd: {
    name: 'recall_cxd',
    description: 'Hybrid semantic + WordNet memory search with CXD filtering',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Search query for semantic + WordNet expansion'
        },
        function_filter: {
          type: 'string',
          description: 'CXD function filter: CONTROL, CONTEXT, DATA, ALL',
          default: 'ALL'
        },
        limit: {
          type: 'number',
          description: 'Maximum results to return',
          default: 5
        },
        db_name: {
          type: 'string',
          description: 'Database to search (memmimic, enhanced, legacy)'
        }
      },
      required: ['query']
    }
  },
  
  // 🧠 MEMORY CORE (3)
  remember: {
    name: 'remember',
    description: 'Store information with automatic CXD classification',
    inputSchema: {
      type: 'object',
      properties: {
        content: {
          type: 'string',
          description: 'Content to remember'
        },
        memory_type: {
          type: 'string',
          description: 'Type of memory (interaction, reflection, milestone)',
          default: 'interaction'
        }
      },
      required: ['content']
    }
  },
  
  think_with_memory: {
    name: 'think_with_memory',
    description: 'Process input with full contextual memory',
    inputSchema: {
      type: 'object',
      properties: {
        input_text: {
          type: 'string',
          description: 'Input text to process with memory context'
        }
      },
      required: ['input_text']
    }
  },
  
  status: {
    name: 'status',
    description: 'Get MemMimic system status and health',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  
  // 📖 TALES SYSTEM (5)
  tales: {
    name: 'tales',
    description: 'Unified tales interface: list, search, load, and stats',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Search query (if provided, searches tales)'
        },
        stats: {
          type: 'boolean',
          description: 'Show collection statistics',
          default: false
        },
        load: {
          type: 'boolean', 
          description: 'Load tale by name (requires query)',
          default: false
        },
        category: {
          type: 'string',
          description: 'Filter by category (e.g., claude/core, projects/memmimic)'
        },
        limit: {
          type: 'number',
          description: 'Maximum results',
          default: 10
        }
      }
    }
  },
  
  save_tale: {
    name: 'save_tale',
    description: 'Auto-detect create or update tale',
    inputSchema: {
      type: 'object', 
      properties: {
        name: {
          type: 'string',
          description: 'Tale name'
        },
        content: {
          type: 'string',
          description: 'Tale content'
        },
        category: {
          type: 'string',
          description: 'Tale category (claude/core, projects/memmimic, misc/general)',
          default: 'claude/core'
        },
        tags: {
          type: 'string',
          description: 'Comma-separated tags'
        }
      },
      required: ['name', 'content']
    }
  },
  
  load_tale: {
    name: 'load_tale',
    description: 'Load specific tale by name',
    inputSchema: {
      type: 'object',
      properties: {
        name: {
          type: 'string',
          description: 'Name of tale to load'
        },
        category: {
          type: 'string',
          description: 'Specific category to search in'
        }
      },
      required: ['name']
    }
  },
  
  delete_tale: {
    name: 'delete_tale',
    description: 'Delete tale with confirmation',
    inputSchema: {
      type: 'object',
      properties: {
        name: {
          type: 'string',
          description: 'Name of tale to delete'
        },
        category: {
          type: 'string',
          description: 'Specific category to search in'
        },
        confirm: {
          type: 'boolean',
          description: 'Skip confirmation prompt',
          default: false
        }
      },
      required: ['name']
    }
  },
  
  context_tale: {
    name: 'context_tale',
    description: 'Generate narrative from memory fragments',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Story topic (e.g., "introduction", "project history")'
        },
        style: {
          type: 'string',
          description: 'Narrative style (auto, introduction, technical, philosophical)',
          default: 'auto'
        },
        max_memories: {
          type: 'number',
          description: 'Maximum memories to include',
          default: 15
        }
      },
      required: ['query']
    }
  },
  
  // 🔧 ADVANCED FEATURES (2)
  analyze_memory_patterns: {
    name: 'analyze_memory_patterns',
    description: 'Analyze patterns in memory usage and content',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  
  // 🧘 COGNITIVE (1)
  socratic_dialogue: {
    name: 'socratic_dialogue',
    description: 'Engage in Socratic self-questioning',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Topic for Socratic analysis'
        },
        depth: {
          type: 'number',
          description: 'Depth of questioning (1-5)',
          default: 3
        }
      },
      required: ['query']
    }
  }
};

// Create MCP server
const server = new Server(
  {
    name: 'memmimic',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// List tools handler
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: Object.values(MEMMIMIC_TOOLS)
  };
});

// Call tool handler
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  try {
    log(`Tool called: ${name}`);
    
    switch (name) {
      // 🔍 SEARCH CORE
      case 'recall_cxd': {
        const { query, function_filter = 'ALL', limit = 5, db_name } = args;
        
        const pythonArgs = [query, function_filter, limit.toString()];
        if (db_name) pythonArgs.push('--db', db_name);
        
        const result = await runPythonTool('memmimic_recall_cxd', pythonArgs);
        return { content: [{ type: 'text', text: result }] };
      }
      
      // 🧠 MEMORY CORE
      case 'remember': {
        const { content, memory_type = 'interaction' } = args;
        const result = await runPythonTool('memmimic_remember', [content, memory_type]);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'think_with_memory': {
        const { input_text } = args;
        const result = await runPythonTool('memmimic_think', [input_text]);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'status': {
        const result = await runPythonTool('memmimic_status', []);
        return { content: [{ type: 'text', text: result }] };
      }
      
      // 📖 TALES SYSTEM
      case 'tales': {
        const { query, stats = false, load = false, category, limit = 10 } = args;
        
        const pythonArgs = [];
        if (query) pythonArgs.push(query);
        if (stats) pythonArgs.push('--stats');
        if (load) pythonArgs.push('--load');
        if (category) pythonArgs.push('--category', category);
        if (limit !== 10) pythonArgs.push('--limit', limit.toString());
        
        const result = await runPythonTool('memmimic_tales', pythonArgs);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'save_tale': {
        const { name, content, category = 'claude/core', tags } = args;
        
        const pythonArgs = [name, content, '--category', category];
        if (tags) pythonArgs.push('--tags', tags);
        
        const result = await runPythonTool('memmimic_save_tale', pythonArgs);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'load_tale': {
        const { name, category } = args;
        
        const pythonArgs = [name];
        if (category) pythonArgs.push('--category', category);
        
        const result = await runPythonTool('memmimic_load_tale', pythonArgs);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'delete_tale': {
        const { name, category, confirm = false } = args;
        
        const pythonArgs = [name];
        if (category) pythonArgs.push('--category', category);
        if (confirm) pythonArgs.push('--confirm');
        
        const result = await runPythonTool('memmimic_delete_tale', pythonArgs);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'context_tale': {
        const { query, style = 'auto', max_memories = 15 } = args;
        
        const pythonArgs = [query, '--style', style, '--max-memories', max_memories.toString()];
        
        const result = await runPythonTool('memmimic_context_tale', pythonArgs);
        return { content: [{ type: 'text', text: result }] };
      }
      
      // 🔧 ADVANCED FEATURES
      case 'analyze_memory_patterns': {
        const result = await runPythonTool('memmimic_analyze_patterns', []);
        return { content: [{ type: 'text', text: result }] };
      }
      
      // 🧘 COGNITIVE
      case 'socratic_dialogue': {
        const { query, depth = 3 } = args;
        const result = await runPythonTool('memmimic_socratic', [query, depth.toString()]);
        return { content: [{ type: 'text', text: result }] };
      }
      
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    log(`Error in tool ${name}: ${error.message}`);
    return { 
      content: [{ 
        type: 'text', 
        text: `❌ Error in ${name}: ${error.message}` 
      }],
      isError: true
    };
  }
});

// Error handling
process.on('uncaughtException', (error) => {
  log(`Uncaught Exception: ${error.message}`);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  log(`Unhandled Rejection: ${reason}`);
  process.exit(1);
});

// Main execution
async function main() {
  log('MemMimic MCP Server v1.0 starting...');
  log('11 essential tools, original design restored');
  
  const transport = new StdioServerTransport();
  
  try {
    await server.connect(transport);
    log('✅ MemMimic MCP Server ready!');
  } catch (error) {
    log(`Failed to connect: ${error.message}`);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGINT', () => {
  log('Shutting down gracefully...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  log('Shutting down gracefully...');
  process.exit(0);
});

// Start the server
main().catch((error) => {
  log(`Fatal error: ${error.message}`);
  process.exit(1);
});
