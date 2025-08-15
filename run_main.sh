# Define variables
data_mode='dev' # Options: 'dev', 'train' 
db_root_path=bird #root directory # UPDATE THIS WITH THE PATH TO THE TARGET DATASET
start=0 #闭区间
end=89  #开区间
# pipeline_nodes='generate_db_schema+extract_col_value+extract_query_noun+column_retrieve_and_other_info+candidate_generate+align_correct+vote+evaluation'
pipeline_nodes = 'generate_cosrse_for_filter_column+regenerate_fine+execution_optimization'  # 前面的步骤都是用的别人的
# pipeline指当前工作流的节点组合
 
# Nodes:
    # generate_db_schema
    # extract_col_value
    # extract_query_noun
    # column_retrieve_and_other_info
    # candidate_generate
    # align_correct
    # vote
    # evaluation

AK=$OPENAI_API_KEY #set your ak in src/llm/model.py
engine1='gpt-4o'     #'gpt-4o-0513'
engine2='deepseek-v3'
engine3='gemini-2.5-pro'

pipeline_setup='{
    "generate_cosrse_for_filter_column":{
        "engine": "'${engine1}'",
        "temperature": 1.0,  
        "return_question":"True",
        "single":"False"
    },
    "regenerate_fine":{
        "engine": "'${engine1}'",
        "temperature": 1.0,
        "return_question":"True",
        "single":"False"
    },
    "execution_optimization":{
        "engine": "'${engine1}'",
        "temperature": 1.0,
        "return_question":"True",
        "single":"False"
    }
}'  

# 这竟然还是用langgraph实现的工作流 wq
python3 -u ./main.py --data_mode ${data_mode} --db_root_path ${db_root_path}\
        --pipeline_nodes ${pipeline_nodes} --pipeline_setup "$pipeline_setup"\
        --start ${start} --end ${end} \
        # --use_checkpoint --checkpoint_nodes ${checkpoint_nodes} --checkpoint_dir ${checkpoint_dir}
  
