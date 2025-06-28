FROM node:18-slim

WORKDIR /app

# Install dependencies
RUN npm install express

# Copy source code
COPY server.js .

# Command to run the MCP server
CMD ["node", "server.js"]
