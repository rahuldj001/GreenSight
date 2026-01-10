FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm ci

# Copy source code
COPY . .

# Build the application
# We accept the API URL as a build argument
ARG NEXT_PUBLIC_API_URL=http://localhost:8501
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# Expose the port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]
