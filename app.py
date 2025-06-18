# ========== TAB 4: VIEW FEEDBACK HISTORY ==========
with tab4:
    st.write("## 📋 User Feedback History")
    user_df, tools_df = load_data_fresh()
    tools_df['Title_clean'] = tools_df['Title'].str.lower().str.strip()

    feedback_df = pd.read_sql_query("SELECT * FROM feedback", conn)

    if feedback_df.empty:
        st.info("No feedback has been recorded yet.")
    else:
        selected_feedback_user = st.selectbox("🔍 Select a User to View Feedback", feedback_df['userID'].unique())

        user_feedback = feedback_df[feedback_df['userID'] == selected_feedback_user]

        if user_feedback.empty:
            st.warning("This user has not given any feedback yet.")
        else:
            for _, row in user_feedback.iterrows():
                tool_title = row['toolTitle']
                reward = row['reward']

                match_row = tools_df[tools_df['Title_clean'] == tool_title.lower().strip()]
                if not match_row.empty:
                    tool = match_row.iloc[0]
                    st.markdown(f"### [{tool['Title']}]({tool['Title_URL']})")
                    display_resized_image(tool['Image'])

                    if reward > 0:
                        st.success("👍 Marked as Useful")
                    elif reward < 0:
                        st.error("👎 Marked as Not Useful")
                    else:
                        st.info("No opinion recorded")
                else:
                    st.markdown(f"### {tool_title}")
                    st.warning("⚠️ Tool not found in current tool list.")
