FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY server.js .

# Command to run the MCP server
CMD ["node", "server.js"]
