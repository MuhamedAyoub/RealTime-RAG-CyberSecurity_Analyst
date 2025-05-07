FROM fluent/fluentd:v1.16-debian-1
USER root

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ruby-dev && \
    rm -rf /var/lib/apt/lists/*

# Install plugins with compatible versions
RUN fluent-gem install fluent-plugin-docker_metadata_filter
RUN fluent-gem install fluent-plugin-elasticsearch --no-document

# Copy configuration
COPY fluent.conf /fluentd/etc/

# Switch back to fluent user
USER fluent