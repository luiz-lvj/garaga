# This file (along with docker-compose.yml) helps creating reproducible builds
# - It must be manually triggered whenever the Rust code is changed
# - It will update the code generated under src/wasm/pkg
# Usage: docker compose up --build && docker compose down

FROM rust:1.80.1

RUN cargo install wasm-pack

RUN curl -L https://bit.ly/n-install | bash -s -- -y 20.17.0

ENV HOME="/root/"
ENV N_PREFIX="$HOME/n"
ENV PATH="$HOME/n/bin/:$PATH"

WORKDIR /garaga/tools/npm/garaga_ts

# Add a step to output versions
RUN echo "Rust version: $(rustc --version)" && \
    echo "Cargo version: $(cargo --version)" && \
    echo "wasm-pack version: $(wasm-pack --version)" && \
    echo "Node version: $(node --version)" && \
    echo "NPM version: $(npm --version)"

CMD npm ci && npm run build && npm pack && ./fix-chown.sh ../..
