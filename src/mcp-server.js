#!/usr/bin/env node
/**
 * GeoVision MCP Server
 * Advanced Geo-Spy Analysis System with browser control and computer vision
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
  ErrorCode,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as vision from './tools/vision-analyzer.js';
import * as browser from './tools/browser-controller.js';
import * as geo from './tools/geo-engine.js';
import * as control from './tools/computer-control.js';

const execAsync = promisify(exec);

class GeoVisionMCPServer {
  constructor() {
    this.server = new Server(
      {
        name: 'geovision-mcp',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupHandlers();
  }

  setupHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: this.getTools(),
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        return await this.handleToolCall(name, args);
      } catch (error) {
        return {
          content: [{ type: 'text', text: `Error: ${error.message}` }],
          isError: true,
        };
      }
    });
  }

  getTools() {
    return [
      {
        name: 'analyze_image',
        description: 'Analyze an image for geo-location clues',
        inputSchema: {
          type: 'object',
          properties: {
            imagePath: { type: 'string' },
            imageData: { type: 'string' },
          },
        },
      },
      {
        name: 'extract_text',
        description: 'Extract text using OCR',
        inputSchema: {
          type: 'object',
          properties: { imagePath: { type: 'string' } },
          required: ['imagePath'],
        },
      },
      {
        name: 'detect_building_style',
        description: 'Detect architectural style',
        inputSchema: {
          type: 'object',
          properties: { imagePath: { type: 'string' } },
          required: ['imagePath'],
        },
      },
      {
        name: 'analyze_vegetation',
        description: 'Analyze vegetation for region clues',
        inputSchema: {
          type: 'object',
          properties: { imagePath: { type: 'string' } },
          required: ['imagePath'],
        },
      },
      {
        name: 'search_location',
        description: 'Search places by vector similarity',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string' },
            coords: { type: 'string' },
            radius: { type: 'number' },
          },
          required: ['query'],
        },
      },
      {
        name: 'open_google_maps',
        description: 'Open Google Maps with coordinates or query',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string' },
            lat: { type: 'number' },
            lng: { type: 'number' },
            zoom: { type: 'number' },
          },
        },
      },
      {
        name: 'get_current_tab',
        description: 'Get current browser tab',
        inputSchema: { type: 'object', properties: {} },
      },
      {
        name: 'take_screenshot',
        description: 'Take screenshot of screen or window',
        inputSchema: {
          type: 'object',
          properties: {
            fullScreen: { type: 'boolean' },
            outputPath: { type: 'string' },
          },
        },
      },
      {
        name: 'move_mouse',
        description: 'Move mouse cursor',
        inputSchema: {
          type: 'object',
          properties: { x: { type: 'number' }, y: { type: 'number' } },
          required: ['x', 'y'],
        },
      },
      {
        name: 'click',
        description: 'Click at position',
        inputSchema: {
          type: 'object',
          properties: {
            x: { type: 'number' },
            y: { type: 'number' },
            button: { type: 'string', enum: ['left','right','middle'] },
          },
        },
      },
      {
        name: 'type_text',
        description: 'Type text',
        inputSchema: {
          type: 'object',
          properties: { text: { type: 'string' } },
          required: ['text'],
        },
      },
      {
        name: 'geoguess_analyze',
        description: 'Full geoguess pipeline',
        inputSchema: {
          type: 'object',
          properties: {
            imagePath: { type: 'string' },
            description: { type: 'string' },
            nearPark: { type: 'boolean' },
            region: { type: 'string' },
          },
          required: ['imagePath'],
        },
      },
    ];
  }

  async handleToolCall(name, args) {
    switch (name) {
      case 'analyze_image': return await vision.analyzeImage(args.imagePath || args.imageData);
      case 'extract_text': return await vision.extractText(args.imagePath);
      case 'detect_building_style': return await vision.detectBuildingStyle(args.imagePath);
      case 'analyze_vegetation': return await vision.analyzeVegetation(args.imagePath);
      case 'search_location': return await geo.searchLocation(args.query, args.coords, args.radius);
      case 'open_google_maps': return await browser.openGoogleMaps(args);
      case 'get_current_tab': return await browser.getCurrentTab();
      case 'take_screenshot': return await browser.takeScreenshot(args);
      case 'move_mouse': return await control.moveMouse(args.x, args.y);
      case 'click': return await control.click(args.x, args.y, args.button);
      case 'type_text': return await control.typeText(args.text);
      case 'geoguess_analyze': return await this.runGeoGuessAnalysis(args);
      default: throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
    }
  }

  async runGeoGuessAnalysis(args) {
    const results = { step: 'geoguess_analysis', clues: {}, candidates: [] };
    const imageAnalysis = await vision.analyzeImage(args.imagePath);
    results.clues.image = imageAnalysis;
    const text = await vision.extractText(args.imagePath);
    results.clues.text = text;
    const building = await vision.detectBuildingStyle(args.imagePath);
    results.clues.building = building;
    const vegetation = await vision.analyzeVegetation(args.imagePath);
    results.clues.vegetation = vegetation;

    const query = [args.description, text.significant, building.era, building.type, args.region]
      .filter(Boolean).join(' ');
    const searchResults = await geo.searchLocation(query);
    results.candidates = searchResults.results || [];

    return { content: [{ type: 'text', text: JSON.stringify(results, null, 2) }] };
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('GeoVision MCP Server running on stdio');
  }
}

const server = new GeoVisionMCPServer();
server.run().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
