LOCAL_DIR="${LOCAL_DIR:-/usr/xtmp/yh386/CitePretrain/data}"
is_first_time="true" # "true": import data into the database; "false": skip the import step
is_first_time="false" # "true": import data into the database; "false": skip the import step

mkdir -p ${LOCAL_DIR}/database/db
mongod --dbpath ${LOCAL_DIR}/database/db --fork --logpath ${LOCAL_DIR}/database/mongod.log
#mongod --dbpath ${LOCAL_DIR}/database/db



if [ "$is_first_time" = "true" ]; then
    # if it's the first time to import the data, you need to import the data into the database; otherwise, you can skip this step;
    echo "Importing data into the database..."
    mongoimport --db cite_pretrain --collection sciqag --file ${LOCAL_DIR}/knowledge_source/sciqag/docs.jsonl
    mongoimport --db cite_pretrain --collection repliqa --file ${LOCAL_DIR}/knowledge_source/repliqa/docs.jsonl
    mongoimport --db cite_pretrain --collection wikipedia --file ${LOCAL_DIR}/knowledge_source/wikipedia/docs.jsonl
    mongoimport --db cite_pretrain --collection common_crawl --file ${LOCAL_DIR}/knowledge_source/common_crawl/docs.jsonl
    mongoimport --db cite_pretrain --collection gpt --file ${LOCAL_DIR}/knowledge_source/gpt/docs.jsonl
    echo "Data imported into the database successfully."
else
    echo "Data already imported into the database."
fi
