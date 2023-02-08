wait_ready() {
echo "Waiting for the $1 on port $2"
until nc -z $1 $2 ; do
        sleep 1
done
}

wait_ready $LABEL_STUDIO_HOST $LABEL_STUDIO_PORT

# Start the main process.
echo "Starting the main process..."
exec python -u src/main.py